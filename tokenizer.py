import regex as re
import numpy as np
from enum import Enum
from typing import Optional, Union
from collections import Counter, OrderedDict
from itertools import pairwise

class TokenizerBPE(object):
    class SpecialTokens(Enum):
        START   = 256
        END     = 257
        UNKNOWN = 258
        PADDING = 259

        @property
        def str(self):
            return f"<{self.name}>"

        @property
        def byte(self):
            return list(self.str.encode("utf-8"))

        @property
        def id(self):
            return self.value

    def __init__(self, max_bpe: int):
        # Source: ChatGPT4 tokenizer
        # '(?i:[sdmt]|ll|ve|re) -> will group "I've", "I'll", "you're" together (case insensitive)
        # -(?i:me|te|lhes|se|nos|vos|os|as|lhe|[oa]) -> will group "amo-me", "cale-se", "amai-vos" together (case insensitive)
        # [^\r\n\p{L}\p{N}]?+\p{L}+
        #      [^\r\n\p{L}\p{N}]? -> capture chars expect: digits, letters, \n, \r. Example: @, $, !
        #      +\p{L}+ -> capture letters
        #    therefore, capture things like #Python, @gmail
        # \p{N}{1,3} -> group numbers up to 3 digits max
        pattern = r"""'(?i:[sdmt]|ll|ve|re)|-(?i:me|te|lhes|se|nos|vos|os|as|lhe|[oa])|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.pattern = re.compile(pattern)

        self.bpe: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.max_bpe = max_bpe

    def __len__(self):
        return 256 + len(self.SpecialTokens) + len(self.bpe)

    def bytepair_to_chairpair(self):
        return {self.untokenize(b, to_str=True)[0] : t for (b, t) in self.bpe.items()}

    def replace_pair_for_token(self, byte_group: list[int], byte_pair: tuple[int], token: int) -> list[int]:
        if len(byte_group) == 1:
            return byte_group

        new_byte_group = []
        it = iter(range(len(byte_group)))
        for i in it:
            # append the last byte 
            if i == len(byte_group) - 1:
                new_byte_group.append(byte_group[-1])
                continue

            candidate_byte_pair = tuple([byte_group[i], byte_group[i + 1]])
            if candidate_byte_pair == byte_pair:
                new_byte_group.append(token)
                next(it) # skip the next byte
            else:
                new_byte_group.append(byte_group[i])
        return new_byte_group   

    def replace_token_for_pair(self, tokens: list[int], token: int, byte_pair: tuple[int]) -> list[int]:
        untokenized_text = []
        for t in tokens:
            # token -> byte pair
            if t == token:
                untokenized_text.extend(byte_pair)
            else:
                untokenized_text.append(t)
        return untokenized_text

    def compute_tokens(self, texts: Union[str, list[str]]) -> None:
        if isinstance(texts, str):
            texts = [texts,]

        texts_join = " ".join(texts)
        # text: isto não faz sentido
        # findall: ["isto", " não", " faz", " sentido"]
        # encode: [b"isto", b" n\xc3\xa3o", b" faz", b" sentido"]
        # list: [[105, 115, 116, 111], [32, 110, 195, 163, 111], [32, 102, 97, 122], [32, 115, 101, 110, 116, 105, 100, 111]]
        groups_utf8 = [list(group.encode("utf-8")) for group in self.pattern.findall(texts_join)]

        for _ in range(self.max_bpe):
            bp_count = Counter()
            new_token = self.__len__()

            # find most occurent pair
            bp_count.update([byte_pair for byte_group in groups_utf8 for byte_pair in pairwise(byte_group)])
            bp_most_frequent, count = bp_count.most_common(1)[0]
            self.bpe[bp_most_frequent] = new_token

            # replace the most frequent bytepair by the new token
            for byte_group_idx, byte_group in enumerate(groups_utf8):
                groups_utf8[byte_group_idx] = self.replace_pair_for_token(byte_group, bp_most_frequent, new_token)

    def tokenize(self, texts: Union[str, list[str]]) -> list[list[int]]:
        if isinstance(texts, str):
            texts = [texts,]

        tokenized_texts = []
        for text in texts:
            text_bytes = list(text.encode("utf-8"))
            text_tokens = text_bytes
            for byte_pair, new_token in self.bpe.items():
                text_tokens = self.replace_pair_for_token(text_tokens, byte_pair, new_token)
            tokenized_texts.append(text_tokens)
        return tokenized_texts

    def untokenize(self, tokenized_texts: Union[list[int], list[list[int]]], to_str: bool = False) -> list[str]:
        if isinstance(tokenized_texts, (list, tuple)) and isinstance(tokenized_texts[0], int):
            tokenized_texts = [tokenized_texts,]

        untokenized_texts = []
        for tokenized_text in tokenized_texts:
            # iterate through every byte pair encoding (last to first added)
            for byte_pair, new_token in sorted(self.bpe.items(), reverse=True):
                untokenized_text = self.replace_token_for_pair(tokenized_text, new_token, byte_pair)
                tokenized_text = untokenized_text

            untokenized_texts.append(untokenized_text)

        return [bytes(t).decode("utf-8") for t in untokenized_texts] if to_str else untokenized_texts

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

class TokenizerWords(object):
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
    
    tokenizer = TokenizerWords()
    tokenizer.compute_tokens(texts)
    tokenids = tokenizer.tokenize(texts)

    print("Token IDs")
    for id in tokenids[:3]:
        print(id)
    print()

    print("Padded token IDs")
    for padded in tokenizer.pad(tokenids[:3]):
        print(padded)
    print()

    print("Unpadded token IDs")
    for unpadded in tokenizer.unpad(tokenids[:3]):
        print(unpadded)
    print()

    texts = ["Palhaço verde veste azul nos finais de semana? Isso não faz sentido! É uma loucura..."]

    tokenizer = TokenizerBPE(max_bpe=10)
    tokenizer.compute_tokens(texts)

    text = texts[0]
    text_original_bytes = list(text.encode("utf-8"))
    print("Original:", len(text_original_bytes))
    print("Text:", text)
    print("Bytes:", text_original_bytes)
    print()

    text_tokenized = tokenizer.tokenize(text)[0]
    print("Tokenized:", len(text_tokenized))
    print("Bytes:", text_tokenized)
    print("BPE", tokenizer.bytepair_to_chairpair())
    print()

    text_tokenized_padded = tokenizer.pad(text_tokenized, max_len=80)[0]
    print("Padded:", len(text_tokenized_padded))
    print("Bytes:", text_tokenized_padded)
    print()

    text_tokenized_unpadded = tokenizer.unpad(text_tokenized_padded)[0]
    print("Unpadded:", len(text_tokenized_unpadded))
    print("Bytes:", text_tokenized_unpadded)
    print()    

    text_untokenized = tokenizer.untokenize(text_tokenized, to_str=True)[0]
    print("Untokenized:", len(list(text_untokenized.encode("utf-8"))))
    print("Text:", text_untokenized)