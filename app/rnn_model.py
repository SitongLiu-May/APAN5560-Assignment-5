from typing import List, Dict, Tuple
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, corpus: List[str], word_to_idx: Dict[str, int], seq_len: int = 5):
        self.seq_len = seq_len
        self.word_to_idx = word_to_idx
        self.samples: List[Tuple[List[int], int]] = []

        for sentence in corpus:
            tokens = sentence.lower().strip().split()
            if len(tokens) <= 1:
                continue
            idxs = [word_to_idx[w] for w in tokens if w in word_to_idx]
            if len(idxs) <= 1:
                continue

            for i in range(1, len(idxs)):
                start = max(0, i - seq_len)
                context = idxs[start:i]
                target = idxs[i]
                # left pad with BOS (index 0) if context too short
                if len(context) < seq_len:
                    context = [0] * (seq_len - len(context)) + context
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        # use last hidden state
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


class RNNModel:
    """
    High-level wrapper that:
    - builds vocabulary from corpus
    - trains a small LSTM language model
    - exposes generate(start_word, length)
    """

    def __init__(self, corpus: List[str], seq_len: int = 5, epochs: int = 10, lr: float = 1e-2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Build vocabulary (index 0 reserved for padding/BOS)
        vocab = {"<PAD>": 0}
        for sentence in corpus:
            for w in sentence.lower().strip().split():
                if w not in vocab:
                    vocab[w] = len(vocab)

        self.word_to_idx = vocab
        self.idx_to_word = {i: w for w, i in vocab.items()}
        vocab_size = len(vocab)

        dataset = TextDataset(corpus, self.word_to_idx, seq_len=seq_len)
        self.model = LSTMModel(vocab_size).to(device)

        if len(dataset) == 0:
            # nothing to train, just keep random model
            return

        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits = self.model(x)
                loss = criterion(logits, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.item()

    def _encode_word(self, word: str) -> int:
        word = word.lower()
        return self.word_to_idx.get(word, 0)

    def _decode_word(self, idx: int) -> str:
        return self.idx_to_word.get(idx, "<UNK>")

    def generate(self, start_word: str, length: int) -> str:

        if length <= 0:
            return ""

        self.model.eval()
        words = [start_word.lower()]

        with torch.no_grad():
            for _ in range(length - 1):
                context = [self._encode_word(w) for w in words[-5:]]
                if len(context) < 5:
                    context = [0] * (5 - len(context)) + context
                x = torch.tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                # greedy
                next_idx = int(torch.argmax(probs).item())
                next_word = self._decode_word(next_idx)
                words.append(next_word)

        return " ".join(words)
