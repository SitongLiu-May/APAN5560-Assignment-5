import random
from collections import defaultdict, Counter
from typing import List, Dict


class BigramModel:

    def __init__(self, corpus: List[str]):
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.vocab = set()

        for sentence in corpus:
            words = self._tokenize(sentence)
            if not words:
                continue
            self.unigram_counts.update(words)
            self.vocab.update(words)

            # build bigram counts
            for w1, w2 in zip(words[:-1], words[1:]):
                self.bigram_counts[w1][w2] += 1

        self.vocab = sorted(list(self.vocab))
        # precompute unigram distribution for backoff
        self._unigram_total = sum(self.unigram_counts.values())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # simple whitespace tokenization + lowercase
        return text.lower().strip().split()

    def _sample_from_counter(self, counter: Counter) -> str:
        total = sum(counter.values())
        r = random.uniform(0, total)
        cum = 0.0
        for word, count in counter.items():
            cum += count
            if r <= cum:
                return word
        # fallback (should not happen)
        return random.choice(list(counter.keys()))

    def generate_text(self, start_word: str, length: int) -> str:
        if length <= 0:
            return ""

        start_word = start_word.lower()
        words = [start_word]

        for _ in range(length - 1):
            prev = words[-1]

            if prev in self.bigram_counts and self.bigram_counts[prev]:
                next_word = self._sample_from_counter(self.bigram_counts[prev])
            else:
                # fall back to unigram distribution
                next_word = self._sample_from_counter(self.unigram_counts)

            words.append(next_word)

        return " ".join(words)
