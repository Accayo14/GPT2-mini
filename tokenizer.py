from __future__ import annotations

import tiktoken


class GPT2Tokenizer:
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self.enc.eot_token
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self.enc.encode_ordinary(text)
        if add_eot:
            ids.append(self.eot_token)
        return ids

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
