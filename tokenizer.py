import re
import numpy as np
from enum import Enum
from typing import Optional, Union

class Tokenizer(object):
    class SpecialTokens(Enum):
        START   = 0
        END     = 1
        UNKNOWN = 2
        PADDING = 3

        @property
        def str(self):
            return f"<{self.name}>"
        
        @property
        def id(self):
            return self.value

    def __init__(self):
        self.pattern = r"\w+(?:'\w+)+|\w+(?:-\w+)+|\w+|[,.!?]|[^\w\s]"
        self.token2tokenid = {}
        self.tokenid2token = {}

    def __len__(self):
        return len(self.token2tokenid)

    def compute_tokens(self, texts: list[str]) -> None:
        # concatenate all texts together and transform them to lowercase
        concatenated_text = " ".join(texts)
        # split text into words and special characters, i.e., tokens
        tokens = re.findall(self.pattern, concatenated_text)
        # find unique tokens and return their counts
        unique_tokens, unique_tokens_count = np.unique(tokens, return_counts=True)

        # unique tokens sorted in descending order (high to low frequency)
        sort_count_indices = np.argsort(unique_tokens_count)[::-1]
        # sort unique tokens from high to low frequency
        unique_tokens = unique_tokens[sort_count_indices].tolist()

        num_special_tokens = len(self.SpecialTokens)
        tokenids = range(num_special_tokens, num_special_tokens + len(unique_tokens))
        # associating tokens to id. The id number is inversely proportional to the token frequency
        self.token2tokenid = {str(token) : tokenid for token, tokenid in zip(unique_tokens, tokenids)}

        # adding special tokens
        for special_token in self.SpecialTokens:
            self.token2tokenid[special_token.str] = special_token.id

        self.tokenid2token = {val : key for key, val in self.token2tokenid.items()}

    def tokenize(self, texts: Union[str, list[str]]) -> list[list[int]]:
        if isinstance(texts, str):
            texts = [texts,]

        tokenized_texts = []
        for text in texts:
            tokens = re.findall(self.pattern, text)
            tokenized_text = [self.token2tokenid.get(token, self.SpecialTokens.UNKNOWN.id) for token in tokens]
            tokenized_texts.append(tokenized_text)

        return tokenized_texts

    def untokenize(self, tokenized_texts: Union[list[int], list[list[int]]], join: bool = False) -> list[str]:
        if isinstance(tokenized_texts, list) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        untokenized_texts = [[self.tokenid2token.get(token, self.SpecialTokens.UNKNOWN.str) for token in tokenized_text]
                             for tokenized_text in tokenized_texts]

        if join:
            return [" ".join(untokenized_text) for untokenized_text in untokenized_texts]
        
        return untokenized_texts

    @classmethod
    def pad(cls, tokenized_texts: Union[list[int], list[list[int]]], max_len: Optional[int] = None) -> list[list[int]]:
        if isinstance(tokenized_texts, list) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        if max_len is None:
            # find the maximum length among the texts
            max_len = max(map(len, tokenized_texts))

        padded = []
        for tt in tokenized_texts:
            pad_num = max_len - len(tt)
            ptt = tt + [cls.SpecialTokens.PADDING.id,] * pad_num * (pad_num > 0)
            padded.append(ptt)

        return padded

    @classmethod
    def pad_mask(cls, tokenized_texts: Union[list[int], list[list[int]]]) -> list[list[bool]]:
        if isinstance(tokenized_texts, list) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        mask = []
        for tt in tokenized_texts:
            try:
                num_false = tt.index(cls.SpecialTokens.PADDING.id)
                num_true  = len(tt) - num_false
            # no padding found
            except ValueError:
                num_false = len(tt)
                num_true = 0

            mask.append([False,] * num_false + [True,] * num_true)

        return mask

    @classmethod
    def unpad(cls, tokenized_texts: Union[list[int], list[list[int]]]) -> list[list[int]]:
        if isinstance(tokenized_texts, list) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        unpadded = []
        for tt in tokenized_texts:
            try:
                padding_start_idx = tt.index(cls.SpecialTokens.PADDING.id)
            # no padding found
            except ValueError:
                padding_start_idx = None

            unpadded.append(tt[:padding_start_idx])

        return unpadded

    @classmethod
    def cap_after_end(cls, tokenized_texts: Union[list[int], list[list[int]]]) -> list[list[int]]:
        if isinstance(tokenized_texts, list) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        capped = []
        for tt in tokenized_texts:
            try:
                end_idx = tt.index(cls.SpecialTokens.END.id)
                capped.append(tt[:end_idx])
            # no end found
            except ValueError:
                capped.append([])

        return capped

if __name__ == "__main__":
    tokenizer = Tokenizer()
    texts = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
             "Donec eu arcu eget velit ornare bibendum.",
             "Nulla convallis lectus a lectus imperdiet, sed dignissim lacus lobortis.",
             "Pellentesque convallis lorem et accumsan aliquam.",
             "Mauris et felis accumsan, fermentum felis vitae, efficitur urna.",
             "Duis placerat ex eget tortor pellentesque consectetur.",
             "Aliquam ut sem non erat mattis porta nec quis arcu.",
             "Nulla sed elit efficitur, bibendum magna non, consequat augue.",
             "Suspendisse in elit vel quam sodales maximus sit amet tempor ante.",
             "Praesent congue orci eget pharetra convallis.",
             "Morbi faucibus velit eget elit cursus commodo.",
             "Suspendisse ut felis a enim suscipit lobortis vitae ac ante.",
             "Proin accumsan orci in orci suscipit, sit amet lobortis augue placerat.",
             "Proin fringilla erat non dui rutrum efficitur.",
             "Suspendisse consequat dolor quis nisi facilisis tempus.",
             "Aenean congue sapien in egestas pulvinar.",]
    
    tokenizer.compute_tokens(texts)
    tokenids = tokenizer.tokenize(texts)

    print("Token IDs")
    for id in tokenids[:3]:
        print(id)

    print("Padded token IDs")
    for padded in tokenizer.pad(tokenids[:3]):
        print(padded)

    print("Unpadded token IDs")
    for unpadded in tokenizer.unpad(tokenids[:3]):
        print(unpadded)