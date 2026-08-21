import torch
from tokenizer import TokenizerBPE

class CollateFn(object):
    def __init__(self, tokenizer: TokenizerBPE):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        source_list = []
        target_list = []
        source_tokenized_list = []
        target_tokenized_list = []
        for (source, target) in batch:
            source_tokenized = self.tokenizer.tokenize(source)[0]

            target_tokenized  = [TokenizerBPE.SpecialTokens.START.id,]
            target_tokenized += self.tokenizer.tokenize(target)[0]
            target_tokenized += [TokenizerBPE.SpecialTokens.END.id,]

            source_list.append(source)
            target_list.append(target)
            source_tokenized_list.append(source_tokenized)
            target_tokenized_list.append(target_tokenized)

        max_len_source = max(map(len, source_tokenized_list))
        max_len_target = max(map(len, target_tokenized_list))
        
        source_tokenized_padded = self.tokenizer.pad(source_tokenized_list, max_len_source)
        target_tokenized_padded = self.tokenizer.pad(target_tokenized_list, max_len_target)

        source_pad_mask = self.tokenizer.pad_mask(source_tokenized_padded)
        target_pad_mask = self.tokenizer.pad_mask(target_tokenized_padded)

        return torch.tensor(source_tokenized_padded, dtype=torch.long), \
               torch.tensor(source_pad_mask, dtype=bool), \
               torch.tensor(target_tokenized_padded, dtype=torch.long), \
               torch.tensor(target_pad_mask, dtype=bool), \
               source_list, \
               target_list