#!/usr/bin/env python3
"""Rebuild the E2E vocabulary exactly as Diffusion-LM training does."""

import json
import sys
from collections import Counter
from pathlib import Path

from spacy.lang.en import English


source, output = map(Path, sys.argv[1:])
tokenizer = English().tokenizer
counts = Counter()

with source.open(encoding="utf-8") as handle:
    for line in handle:
        fields = line.split("||")
        if len(fields) < 2:
            raise ValueError(f"invalid E2E row: {line!r}")
        counts.update(token.text for token in tokenizer(fields[1]))

vocab = {"START": 0, "END": 1, "UNK": 2, "PAD": 3}
for token, count in counts.items():
    if count > 10:
        vocab[token] = len(vocab)

output.write_text(json.dumps(vocab), encoding="utf-8")
print(f"Wrote {len(vocab)} tokens to {output}")
