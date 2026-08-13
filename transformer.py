import torch
import torch.nn as nn
from typing import Optional
from enum import Enum, auto

class HeadType(Enum):
    SELF_ATTENTION = auto()
    MASKED_SELF_ATTENTION = auto()
    CROSS_ATTENTION = auto()

class Head(nn.Module):
    def __init__(self, embedding_size: int, qkv_size: int, device: torch.device, head_type: HeadType):
        super().__init__()
        # projection matrices
        self.Wq = nn.Linear(embedding_size, qkv_size, device=device)
        self.Wk = nn.Linear(embedding_size, qkv_size, device=device)
        self.Wv = nn.Linear(embedding_size, qkv_size, device=device)
        self.qkv_size_sqrt = torch.tensor(qkv_size).sqrt()

        self.head_type = head_type

    #NOTE: for cross attention X1 is the decoder input and X2 is the encoder output
    def forward(self, 
                X1: torch.Tensor, 
                X1_pad_mask: torch.Tensor, 
                X2: Optional[torch.Tensor] = None, 
                X2_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.head_type in [HeadType.SELF_ATTENTION, HeadType.MASKED_SELF_ATTENTION]:
            q, k, v = self.kqv_for_self_attention(X1)
        elif self.head_type == HeadType.CROSS_ATTENTION:
            q, k, v = self.qkv_for_cross_attention(X1, X2)

        # (q @ k.T) / sqrt(qkv_size)
        # NOTE: tranposing last two dims for batched matmul
        qk = torch.matmul(q, k.transpose(1, 2)) / self.qkv_size_sqrt

        # mask the padding tokens so they don't contribute to the attention scores
        if self.head_type in [HeadType.SELF_ATTENTION, HeadType.MASKED_SELF_ATTENTION]:
            # qk = qk.masked_fill(X1_pad_mask.unsqueeze(2), -torch.inf)
            qk = qk.masked_fill(X1_pad_mask.unsqueeze(1), -torch.inf)
        elif self.head_type == HeadType.CROSS_ATTENTION:
            # qk = qk.masked_fill(X1_pad_mask.unsqueeze(2), -torch.inf)
            qk = qk.masked_fill(X2_pad_mask.unsqueeze(1), -torch.inf)

        # masked attention: set the values above the diagonal to -inf to reduce their score
        # in the softmax, essentially not allowing the current word to peek the next        
        # Example:
        #   [[1, -inf, -inf],
        #    [2,    3, -inf],
        #    [4,    5,    6]]
        if self.head_type == HeadType.MASKED_SELF_ATTENTION:
            seq_len: int = X1.size(1)
            mask = torch.triu(torch.full((1, seq_len, seq_len), True, device=X1.device), diagonal=1)
            qk = qk.masked_fill(mask, -torch.inf)

        # Example:
        #                   -> [[w1 x w1], [w1 x w2], ... [w1 x wn],
        # softmax direction ->  [w2 x w1], [w2 x w2], ... [w2 x wn],
        #                       ...
        #                   ->  [wn x w1], [wn x w2], ... [wn x wn]]
        # 
        # wi x wj represents the correlation of the ith to the jth word in logits
        attention_scores = torch.softmax(qk, dim=2)
        
        attention = torch.matmul(attention_scores, v)
        return attention

    def kqv_for_self_attention(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        q: torch.Tensor = self.Wq(X) # X @ Wq
        k: torch.Tensor = self.Wk(X)
        v: torch.Tensor = self.Wv(X)
        return q, k, v

    def qkv_for_cross_attention(self, X1: torch.Tensor, X2: torch.Tensor) -> tuple[torch.Tensor]:
        q: torch.Tensor = self.Wq(X1) # X @ Wq
        k: torch.Tensor = self.Wk(X2)
        v: torch.Tensor = self.Wv(X2)
        return q, k, v

class MultiHead(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, device: torch.device, head_type: HeadType):
        super().__init__()
        assert embedding_size % num_heads == 0, "Embedding size should be divisible by the number of heads."

        # projection matrix
        self.Wo = nn.Linear(embedding_size, embedding_size, device=device)

        qkv_size: int = embedding_size // num_heads
        self.heads = nn.ModuleList([Head(embedding_size, qkv_size, device, head_type) for _ in range(num_heads)])

        self.qkv_size = qkv_size
        self.embedding_size = embedding_size

    def forward(self, 
                X1: torch.Tensor, 
                X1_pad_mask: torch.Tensor, 
                X2: Optional[torch.Tensor] = None, 
                X2_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size: int = X1.size(0)
        seq_len: int = X1.size(1)

        heads_concat = torch.empty((batch_size, seq_len, self.embedding_size), device=X1.device)
        for head_idx, head in enumerate(self.heads):
            head_slice = slice(head_idx * self.qkv_size, (head_idx + 1) * self.qkv_size)
            heads_concat[..., head_slice] = head(X1, X1_pad_mask, X2, X2_pad_mask)

        # heads_concat @ Wo
        return self.Wo(heads_concat)

class Encoder(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, device: torch.device):
        super().__init__()
        # self attention
        self.multihead = MultiHead(num_heads, embedding_size, device, HeadType.SELF_ATTENTION)
        self.dropout_multihead = nn.Dropout(0.1, inplace=True)            
        self.norm_multihead = nn.LayerNorm(embedding_size, device=device)
        # feedforward
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4, device=device),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size * 4 , embedding_size, device=device)
        )
        self.ff_norm = nn.LayerNorm(embedding_size, device=device)

    def forward(self, X: torch.Tensor, X_pad_mask: torch.Tensor) -> torch.Tensor:
        # self attention
        head_out = self.dropout_multihead(self.multihead(X, X_pad_mask))
        head_norm = self.norm_multihead(head_out + X) #  skip connection + norm
        # feedforward
        ff_out = self.ff(head_norm)
        return self.ff_norm(ff_out + head_norm) # skip connection + norm

class Decoder(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, device: torch.device):
        super().__init__()
        # self attention masked
        self.multihead_masked = MultiHead(num_heads, embedding_size, device, HeadType.MASKED_SELF_ATTENTION)
        self.dropout_multihead_masked = nn.Dropout(0.1, inplace=True)
        self.norm_multihead_masked = nn.LayerNorm(embedding_size, device=device)
        # cross attention
        self.multihead = MultiHead(num_heads, embedding_size, device, HeadType.CROSS_ATTENTION)
        self.dropout_multihead = nn.Dropout(0.1, inplace=True)
        self.norm_multihead = nn.LayerNorm(embedding_size, device=device)
        # feedforward
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4, device=device),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size * 4, embedding_size, device=device)
        )        
        self.ff_norm = nn.LayerNorm(embedding_size, device=device)

    def forward(self, X1: torch.Tensor, X1_pad_mask: torch.Tensor, X2: torch.Tensor, X2_pad_mask: torch.Tensor) -> torch.Tensor:
        # self attention masked
        head_masked_out = self.dropout_multihead_masked(self.multihead_masked(X1, X1_pad_mask))
        head_masked_norm = self.norm_multihead_masked(head_masked_out + X1) # skip connection + norm
        # cross attention
        head_out = self.dropout_multihead(self.multihead(head_masked_norm, X1_pad_mask, X2, X2_pad_mask))
        head_norm = self.norm_multihead(head_out + head_masked_norm) # skip connection + norm
        # feedforward
        ff_out = self.ff(head_norm)
        return self.ff_norm(ff_out + head_norm) # skip connection + norm

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, max_len: int, device: torch.device):
        super().__init__()
        pe_div = torch.pow(torch.tensor(1e4), torch.arange(0, embedding_size, 2) / embedding_size) 
        pe_matrix = torch.arange(max_len).reshape(-1, 1).repeat(1, embedding_size).float()

        pe_matrix[:, 0::2] = torch.sin(pe_matrix[:, 0::2] / pe_div)
        pe_matrix[:, 1::2] = torch.cos(pe_matrix[:, 1::2] / pe_div)

        pe_matrix = pe_matrix.to(device)
        self.register_buffer("pe_matrix", pe_matrix)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        seq_len: int = X.size(1)
        return X + self.pe_matrix[:seq_len]

class Transformer(nn.Module):
    def __init__(self, 
                 num_tokens_X: int, 
                 num_tokens_Y: int,
                 padding_idx: int, 
                 num_encoders: int, 
                 num_decoders: int, 
                 num_heads: int, 
                 embedding_size: int, 
                 device: torch.device):
        super().__init__()
        self.embedding_X = nn.Embedding(num_tokens_X, embedding_size, padding_idx=padding_idx, device=device)
        self.embedding_Y = nn.Embedding(num_tokens_Y, embedding_size, padding_idx=padding_idx, device=device)
        self.positional_encoding = PositionalEncoding(embedding_size, max_len=1024, device=device)
        self.dropout_encoder = nn.Dropout(0.1, inplace=True)
        self.dropout_decoder = nn.Dropout(0.1, inplace=True)
        self.encoders = nn.ModuleList([Encoder(num_heads, embedding_size, device) for _ in range(num_encoders)])
        self.decoders = nn.ModuleList([Decoder(num_heads, embedding_size, device) for _ in range(num_decoders)])
        self.linear = nn.Linear(embedding_size, num_tokens_Y, device=device)
        self.embedding_size_sqrt = torch.tensor(embedding_size).sqrt()

    def forward(self, X: torch.Tensor, X_pad_mask: torch.Tensor, Y: torch.Tensor, Y_pad_mask: torch.Tensor) -> torch.Tensor:
        encoder_in = self.positional_encoding(self.embedding_X(X) * self.embedding_size_sqrt)
        encoder_in = self.dropout_encoder(encoder_in)
        for encoder in self.encoders:
            encoder_out = encoder(encoder_in, X_pad_mask)
            encoder_in = encoder_out

        decoder_in = self.positional_encoding(self.embedding_Y(Y) * self.embedding_size_sqrt)
        decoder_in = self.dropout_decoder(decoder_in)
        for decoder in self.decoders:
            decoder_out = decoder(decoder_in, Y_pad_mask, encoder_out, X_pad_mask)
            decoder_in = decoder_out

        return self.linear(decoder_out)

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

if __name__ == "__main__":
    batch_size = 128
    seq_len = 96
    num_tokens = 2048
    num_encoders = 6
    num_decoders = 6
    num_heads = 8
    embedding_size = 512
    padding_idx = 3

    # device = fetch_available_device()
    device = torch.device("cpu")
    
    X = torch.randint(0, num_tokens, (batch_size, seq_len,), device=device)
    X_pad_mask = torch.randint(0, 2, (batch_size, seq_len,), dtype=bool, device=device)
    Y = torch.randint(0, num_tokens, (batch_size, seq_len,), device=device)
    Y_pad_mask = torch.randint(0, 2, (batch_size, seq_len,), dtype=bool, device=device)

    transformer = Transformer(num_tokens, num_tokens, padding_idx, num_encoders, num_decoders, num_heads, embedding_size, device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.profiler.record_function("model_inference"):
            out: torch.Tensor = transformer(X, X_pad_mask, Y, Y_pad_mask)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    print(f"Transformer output shape {out.shape}, device {out.device}")