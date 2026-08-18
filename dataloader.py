import torch
from tokenizer import TokenizerWords

class CollateFn(object):
    def __init__(self, source_tokenizer: TokenizerWords, target_tokenizer: TokenizerWords):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def __call__(self, batch):
        source_list = []
        target_list = []
        source_tokenized_list = []
        target_tokenized_list = []
        for (source, target) in batch:
            source_tokenized = self.source_tokenizer.tokenize(source)[0]

            target_tokenized  = [TokenizerWords.SpecialTokens.START.id,]
            target_tokenized += self.target_tokenizer.tokenize(target)[0]
            target_tokenized += [TokenizerWords.SpecialTokens.END.id,]

            source_list.append(source)
            target_list.append(target)
            source_tokenized_list.append(source_tokenized)
            target_tokenized_list.append(target_tokenized)

        max_len_source = max(map(len, source_tokenized_list))
        max_len_target = max(map(len, target_tokenized_list))
        
        source_tokenized_padded = TokenizerWords.pad(source_tokenized_list, max_len_source)
        target_tokenized_padded = TokenizerWords.pad(target_tokenized_list, max_len_target)

        source_pad_mask = TokenizerWords.pad_mask(source_tokenized_padded)
        target_pad_mask = TokenizerWords.pad_mask(target_tokenized_padded)

        return torch.tensor(source_tokenized_padded, dtype=torch.long), \
               torch.tensor(source_pad_mask, dtype=bool), \
               torch.tensor(target_tokenized_padded, dtype=torch.long), \
               torch.tensor(target_pad_mask, dtype=bool), \
               source_list, \
               target_list