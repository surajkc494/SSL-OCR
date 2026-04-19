from __future__ import annotations

from typing import List


class TeluguVocab:
    def __init__(self, vocab_file: str):
        self.char2idx = {}
        self.idx2char = {}
        with open(vocab_file, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                grapheme, idx = parts[0], int(parts[1])
                self.char2idx[grapheme] = idx
                self.idx2char[idx] = grapheme

        self.vocab_size = len(self.char2idx)
        self.blank_idx = self.vocab_size
        self._sorted_keys = sorted(self.char2idx.keys(), key=len, reverse=True)

    def encode(self, word: str) -> List[int]:
        """Tokenize a Telugu word into vocab index list using greedy longest match."""
        tokens = []
        i = 0
        while i < len(word):
            matched = False
            for key in self._sorted_keys:
                if word[i:i + len(key)] == key:
                    tokens.append(self.char2idx[key])
                    i += len(key)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Unknown grapheme at position {i} in word '{word}': remaining='{word[i:]}'"
                )
        return tokens

    def decode(self, indices: List[int]) -> str:
        """Convert index list to Telugu string (skips blank_idx)."""
        return ''.join(
            self.idx2char[i] for i in indices if i != self.blank_idx and i in self.idx2char
        )

    def ctc_decode(self, indices: List[int]) -> str:
        """CTC decode: collapse repeats, remove blanks, decode to string."""
        collapsed = []
        prev = None
        for i in indices:
            if i != prev:
                collapsed.append(i)
            prev = i

        no_blank = [i for i in collapsed if i != self.blank_idx]
        return self.decode(no_blank)
