import pickle
import regex as re
from typing_extensions import Iterable

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.vocab_id: dict[bytes, int] = {self.vocab[i]:i for i in self.vocab}
        self.merges = merges
        self.special_tokens = special_tokens
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pretoken_byte:list[bytes] = sum([[bytes([i]) for i in list(x.encode("utf-8"))] for x in pretoken], [])
        for merge in self.merges:
            index = 0
            while index+1 < len(pretoken_byte):
                if tuple(pretoken_byte[index:index+2]) == merge:
                    pretoken_byte = pretoken_byte[:index]+[pretoken_byte[index]+pretoken_byte[index+1]]+pretoken_byte[index+2:]
                index += 1
        return [self.vocab_id[i] for i in pretoken_byte]
    def encode(self, text: str) -> list[int]:
        PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat_re = re.compile(PAT)
        id_list:list[int] = []
        if self.special_tokens:
            special_tokens = sorted(set(self.special_tokens), key=len, reverse=True)
            special_set = set(special_tokens)
            split_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
            parts = re.split(split_pattern, text)
            for part in parts:
                if not part:
                    continue
                if part in special_set:
                    id_list.append(self.vocab_id[part.encode("utf-8")])
                else:
                    for m in pat_re.finditer(part):
                        id_list += self._encode_pretoken(m.group(0))
        else:
            for m in pat_re.finditer(text):
                                id_list += self._encode_pretoken(m.group(0))
        return id_list
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self, ids: list[int]) -> str:
        # str_bytes = sum([self.vocab[i] for i in ids],b'')
        str_bytes = b''.join([self.vocab[i] for i in ids])
        return str_bytes.decode('utf-8', errors='replace')
