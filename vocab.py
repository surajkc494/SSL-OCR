from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import List, Tuple


class TeluguVocab:
    def __init__(self, vocab_file: str, normalization: str = 'NFC'):
        self.normalization = normalization
        self.char2idx = {}
        self.idx2char = {}

        with open(vocab_file, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                grapheme, idx = self.normalize_text(parts[0]), int(parts[1])
                self.char2idx[grapheme] = idx
                self.idx2char[idx] = grapheme

        self.vocab_size = len(self.char2idx)
        self.blank_idx = self.vocab_size
        self._sorted_keys = sorted(self.char2idx.keys(), key=len, reverse=True)

    def normalize_text(self, text: str) -> str:
        text = text.replace('\u200c', '')
        return unicodedata.normalize(self.normalization, text)

    def _segment_word(self, word: str, strict: bool = True) -> Tuple[List[int], List[str]]:
        """DP segmentation to reduce failures from overlapping Telugu clusters."""
        word = self.normalize_text(word)

        @lru_cache(maxsize=None)
        def solve(pos: int):
            if pos >= len(word):
                return (0, 0, [])  # unknown_count, token_count, tokens

            best = None
            for key in self._sorted_keys:
                if word[pos : pos + len(key)] == key:
                    unknown_cnt, tok_cnt, toks = solve(pos + len(key))
                    candidate = (unknown_cnt, tok_cnt + 1, [self.char2idx[key]] + toks)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate

            # fallback: skip one codepoint as unknown grapheme slice
            unknown_cnt, tok_cnt, toks = solve(pos + 1)
            fallback = (unknown_cnt + 1, tok_cnt, toks)
            if best is None or fallback[:2] < best[:2]:
                best = fallback

            return best

        unknown_count, _, tokens = solve(0)
        unknowns = []
        if unknown_count > 0:
            # reconstruct unknown segments for debugging
            i = 0
            for _ in range(len(word)):
                matched = False
                for key in self._sorted_keys:
                    if word[i : i + len(key)] == key:
                        i += len(key)
                        matched = True
                        break
                if not matched:
                    unknowns.append(word[i])
                    i += 1
                if i >= len(word):
                    break

            if strict:
                raise ValueError(
                    f"Unknown grapheme(s) in word '{word}': {unknowns}"
                )

        return tokens, unknowns

    def encode(self, word: str, strict: bool = True) -> List[int]:
        tokens, _ = self._segment_word(word, strict=strict)
        return tokens

    def decode(self, indices: List[int]) -> str:
        return ''.join(
            self.idx2char[i] for i in indices if i != self.blank_idx and i in self.idx2char
        )

    def ctc_decode(self, indices: List[int]) -> str:
        collapsed = []
        prev = None
        for i in indices:
            if i != prev:
                collapsed.append(i)
            prev = i

        no_blank = [i for i in collapsed if i != self.blank_idx]
        return self.decode(no_blank)

    @staticmethod
    def _levenshtein_distance(a: List[int], b: List[int]) -> int:
        """Compute edit distance between two token sequences."""
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)

        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (0 if ca == cb else 1)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def edit_distance(self, pred_tokens: List[int], gt_tokens: List[int]) -> int:
        """Levenshtein distance on token-index sequences."""
        return self._levenshtein_distance(pred_tokens, gt_tokens)

    def word_edit_distance(self, pred_word: str, gt_word: str) -> int:
        """
        Word-level edit distance.
        Returns 0 for exact match and 1 for mismatch by design.
        """
        pred_norm = self.normalize_text(pred_word)
        gt_norm = self.normalize_text(gt_word)
        return 0 if pred_norm == gt_norm else 1
