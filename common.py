"""Shared constants and text preprocessing for training and inference."""
import re
from pathlib import Path

VOCAB_SIZE = 10_000
MAXLEN = 200
ART = Path(__file__).resolve().parent / "artifacts"

PAD, START, OOV = 0, 1, 2
_TOKEN = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def encode(words: list[str], vocab: dict[str, int]) -> list[int]:
    return [START] + [vocab.get(w, OOV) for w in words]
