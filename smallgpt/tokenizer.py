import torch


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self._stoi[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._itos[i] for i in tokens)

    def encode_tensor(self, s: str) -> torch.Tensor:
        return torch.tensor(self.encode(s), dtype=torch.long)
