import math
import torch
import random
import argparse
import numpy as np
from typing import Optional
from pathlib import Path
from collections import Counter
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ChatTask2024
from tokenizer import TokenizerBPE
from transformer import Transformer
from dataloader import CollateFn

writer: Optional[SummaryWriter] = None
no_tensorboard: bool = False

def fetch_available_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using device: mps")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using device: cuda")
        return torch.device("cuda")
    else:
        print("Using device: cpu")
        return torch.device("cpu")

def set_determinism():
    # set seed, to be deterministic
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  

# adapted from: https://stackoverflow.com/a/30609050
def find_ngrams(input_list: list[str], n: int) -> set[list[str]]:
    return list(zip(*[input_list[i:] for i in range(n)]))

def safe_div(num, den):
    return num / den if den > 0 else 0

def compute_bleu_score(prediction_token_list: list[list[str]], reference_token_list: list[list[str]]) -> float:
    reference_length_sum = 0
    prediction_length_sum = 0

    for prediction_tokens, reference_tokens in zip(prediction_token_list, reference_token_list):
        reference_length_sum += len(reference_tokens)
        prediction_length_sum += len(prediction_tokens)

    brevity_penalty = math.exp(-max(0, safe_div(reference_length_sum, prediction_length_sum) - 1))

    paired = list(zip(prediction_token_list, reference_token_list))

    nsum = 0.0
    nrange = 4
    smoothing_epsilon = 1e-9

    for n in range(1, nrange + 1):
        precision_numerator = 0
        precision_denominator = 0

        for prediction_tokens, reference_tokens in paired:
            prediction_ngrams = find_ngrams(prediction_tokens, n)
            reference_ngrams = find_ngrams(reference_tokens, n)

            prediction_counts = Counter(prediction_ngrams)
            reference_counts = Counter(reference_ngrams)

            for ngram, count in prediction_counts.items():
                precision_numerator += min(count, reference_counts.get(ngram, 0))
                precision_denominator += count

        precision = safe_div(precision_numerator, precision_denominator)

        precision = max(precision, smoothing_epsilon)
        nsum += math.log(precision) * (1 / nrange)

    return brevity_penalty * math.exp(nsum)

def moving_average(element: float, count: int, mean: float) -> float:
    return (element + count * mean) / (count + 1)

@torch.no_grad()
def val(model: Transformer, val_dataloader: DataLoader, tokenizer: TokenizerBPE, device: torch.device):
    loss = 0.0
    bleu_score = 0.0
    pbar = tqdm(val_dataloader, desc="Validation", leave=False)
    for count, batch in enumerate(pbar, start=1):
        
        source, source_mask, target, target_mask, source_str, target_str = batch

        source = source.to(device) 
        source_mask = source_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        # mask created to skip the special token <END> in the target
        remove_end_token_mask = torch.where(target == TokenizerBPE.SpecialTokens.END.id, False, True)

        batch_size = source.size(0)
        output_logits: torch.Tensor = model(source, 
                                            source_mask, 
                                            target[remove_end_token_mask].reshape(batch_size, -1), 
                                            target_mask[remove_end_token_mask].reshape(batch_size, -1))

        # cross entropy expects input dims: (batch, classes, k1, ..., kn) where ki are any extra dims
        # NOTE: skipping the special token <START> in the target
        batch_loss = torch.nn.functional.cross_entropy(output_logits.transpose(1, 2),
                                                       target[..., 1:],
                                                       ignore_index=TokenizerBPE.SpecialTokens.PADDING.id)        
        
        loss = moving_average(batch_loss.item(), count, loss)

        output_tokens: list[list[int]] = output_logits.argmax(dim=-1).cpu().numpy().tolist()
        output_tokens = tokenizer.cap_after_end(output_tokens)

        output_str_group: list[list[str]] = tokenizer.split_texts(tokenizer.untokenize(output_tokens, to_str=True))
        target_str_group: list[list[str]] = tokenizer.split_texts(target_str)

        batch_bleu_score = compute_bleu_score(output_str_group, target_str_group)
        bleu_score = moving_average(batch_bleu_score, count, bleu_score)

    output_str = ["".join(str_group) for str_group in output_str_group]

    for count, (s, t, o) in enumerate(zip(source_str, target_str, output_str), start=1):
        if count >= 10:
            break
        pbar.write(f"Source: {s}")
        pbar.write(f"Target: {t}")
        pbar.write(f"Output: {o}")

    last_batch_data = [source_str, target_str, output_str]

    return loss, bleu_score, last_batch_data
                                      
def train(model: Transformer, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          train_dataloader: DataLoader,
          device: torch.device,
          epoch: int) -> float:
    loss = 0.0
    pbar = tqdm(train_dataloader, desc="Training", leave=False)
    for count, batch in enumerate(pbar, start=1):

        optimizer.zero_grad()

        source, source_mask, target, target_mask, source_str, target_str = batch
        source = source.to(device)
        source_mask = source_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)        

        # mask created to skip the special token <END> in the target
        remove_end_token_mask = torch.where(target == TokenizerBPE.SpecialTokens.END.id, False, True)

        batch_size = source.size(0)
        output_logits: torch.Tensor = model(source, 
                                            source_mask, 
                                            target[remove_end_token_mask].reshape(batch_size, -1), 
                                            target_mask[remove_end_token_mask].reshape(batch_size, -1))

        # cross entropy expects input dims: (batch, classes, k1, ..., kn) where ki are any extra dims
        # NOTE: skipping the special token <START> in the target
        batch_loss = torch.nn.functional.cross_entropy(output_logits.transpose(1, 2),
                                                       target[..., 1:],
                                                       ignore_index=TokenizerBPE.SpecialTokens.PADDING.id,
                                                       label_smoothing=0.1)

        batch_loss.backward()

        # gradient clipping to avoid large changes in the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # step-based (not epoch-based) lr computation
        scheduler.step()

        loss = moving_average(batch_loss.item(), count, loss)

        step = epoch * len(train_dataloader) + count
        writer.add_scalar("train/loss", batch_loss.item(), step)

    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

    return loss

def train_val_loop(num_epochs: int, 
                   model: Transformer, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LRScheduler,
                   train_dataloader: DataLoader, 
                   val_dataloader: DataLoader,
                   tokenizer: TokenizerBPE,
                   device: torch.device, 
                   eval_every_n_epochs: int):

    val_loss = np.nan
    train_loss = np.nan
    best_bleu_score = 0.0

    pbar = trange(1, num_epochs + 1, desc="Epochs")
    for epoch in pbar:
        model.train()
        train_loss = train(model, optimizer, scheduler, train_dataloader, device, epoch)
        pbar.set_description(f"Losses train: {train_loss:.3f}, val: {val_loss:.3f}")

        if not no_tensorboard:
            torch.save(model.state_dict(), Path(writer.get_logdir()) / "last.pth")

        is_eval = epoch % eval_every_n_epochs == 0
        if not is_eval:
            writer.flush()
            continue

        pbar.write(f"Validation output (epoch {epoch}): ")

        model.eval()
        val_loss, bleu_score, last_batch_data = val(model, val_dataloader, tokenizer, device)

        if bleu_score > best_bleu_score and not no_tensorboard:
            best_bleu_score = bleu_score
            torch.save(model.state_dict(), Path(writer.get_logdir()) / "best.pth")

        is_first_eval = epoch // eval_every_n_epochs == 1
        step = epoch * len(train_dataloader)

        if is_first_eval:
            source_str = [f"{i:>2}: \"{s}\"" for i, s in enumerate(last_batch_data[0][::4], start=1)]
            writer.add_text("eval/source", "\n".join(source_str), step)
            target_str = [f"{i:>2}: \"{s}\"" for i, s in enumerate(last_batch_data[1][::4], start=1)]
            writer.add_text("eval/target", "\n".join(target_str), step)

        prediction_str = [f"{i:>2}: \"{s}\"" for i, s in enumerate(last_batch_data[2][::4], start=1)]
        writer.add_text("eval/prediction", "\n".join(prediction_str), step)

        writer.add_scalar("eval/loss", val_loss, step)
        writer.add_scalar("eval/bleu", bleu_score, step)
        writer.flush()

def set_tensorboard(experiment_name: Optional[str] = None):
    global writer
    if no_tensorboard:
        class DummyWriter:
            def __getattr__(self, name):
                # Silently ignores all .add_scalar, .add_image, etc.
                return lambda *args, **kwargs: None
        writer = DummyWriter()
        print("TensorBoard logging is DISABLED.")
    else:
        if experiment_name is not None:
            writer = SummaryWriter(Path("runs") / experiment_name)
        else:
            writer = SummaryWriter()
        print("TensorBoard logging is ENABLED.")    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", "-d", type=str, default="./chat-task-2024-data/",
                        help="Path to the WMT 2024 Chat task dataset.")      
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--eval-every-n-epochs", "-v", type=int, default=5,
                        help="Validate every n epochs.")
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help="Batch size used for training. The higher the better but slower.")
    parser.add_argument("--embedding-size", "-k", type=int, default=256,
                        help="Embedding size used for training. The higher the better but slower.")
    parser.add_argument("--lr", "-l", type=float, default=1e-4,
                        help="Learning rate used for training. "
                             "If too high the model weights might take steep steps during optimization.")  
    parser.add_argument("--first-n-samples", "-f", type=int, default=-1,
                        help="Train using only the first n samples. Useful for debugging.")
    parser.add_argument("--load-model-path", "-p", type=str, default="",
                        help="Load the model weights from the provided path.")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name of the experiment.") 
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard.")
    parser.add_argument("--train-is-val", action="store_true",
                        help="The validation set is the training set. Useful for debugging (is the model learning?).")
    
    return parser.parse_args()

def main():
    args = get_args()

    global no_tensorboard
    no_tensorboard = args.no_tensorboard
    set_tensorboard(args.experiment_name)

    set_determinism()
    
    # debugging to check if nan pollutes the model
    # torch.autograd.set_detect_anomaly(True)

    train_dataset = ChatTask2024(args.dataset_root, split="train", source_language="en", first_n_samples=args.first_n_samples)
    if args.train_is_val:
        val_dataset = ChatTask2024(args.dataset_root, split="train", source_language="en", first_n_samples=args.first_n_samples)
    else:
        val_dataset = ChatTask2024(args.dataset_root, split="valid", source_language="en", first_n_samples=args.first_n_samples)

    tokenizer: TokenizerBPE = None
    tokenizer_path = Path("tokenizer_bpe.pkl")
    if tokenizer_path.is_file() and tokenizer_path.suffix == ".pkl":
        tokenizer = TokenizerBPE.load(tokenizer_path)
    else:
        tokenizer = TokenizerBPE(max_bpe=2000)
        tokenizer.compute_tokens(np.hstack((train_dataset.source, train_dataset.target)))
        tokenizer.save(tokenizer_path)

        from pprint import pprint
        pprint(tokenizer.bytepair_to_chairpair(step=10))

    device = fetch_available_device()

    num_heads = 8
    num_encoders = 6
    num_decoders = 6
    padding_idx = TokenizerBPE.SpecialTokens.PADDING.id
    num_tokens = len(tokenizer)
    model = Transformer(num_tokens, 
                        padding_idx, 
                        num_encoders, 
                        num_decoders, 
                        num_heads, 
                        args.embedding_size, 
                        device)

    load_model_path = Path(args.load_model_path)
    if load_model_path.is_file() and load_model_path.suffix == ".pth":
        print(f"Loading model from path: {load_model_path}")
        model.load_state_dict(torch.load(load_model_path, map_location=device))

    collate_fn = CollateFn(tokenizer)

    num_workers = 0
    train_dataloader = DataLoader(train_dataset, 
                                  args.batch_size, 
                                  num_workers=num_workers,
                                  persistent_workers=(num_workers > 0),
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, 
                                args.batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn)    

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.98), eps=1e-9)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    decay_steps = total_steps - warmup_steps
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=args.lr * 0.01)
    # switch after warmup steps
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    train_val_loop(args.epochs, 
                   model, 
                   optimizer, 
                   scheduler, 
                   train_dataloader, 
                   val_dataloader, 
                   tokenizer, 
                   device, 
                   args.eval_every_n_epochs)

if __name__ == "__main__":
    main()