"""
Нейросеть: посимвольная двунаправленная LSTM (CharBiLSTM).

Вход:   слово как последовательность букв        к р у ж к а м и
Выход:  вероятность каждого из ~6700 классов      (тег + правило леммы)

Слои:
  1. Embedding  — каждая буква -> вектор из 64 чисел (буквы с похожей
                  "ролью" получают похожие векторы).
  2. BiLSTM     — читает слово слева направо и справа налево и
                  накапливает "память" о прочитанном. 2 слоя.
  3. Linear+ReLU+Dropout — смешивает признаки.
  4. Linear     — по одному числу (логиту) на класс; softmax -> вероятности.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAX_LEN = 24          # длиннее — берём последние 24 буквы (важен конец слова)


class CharBiLSTM(nn.Module):
    def __init__(self, n_chars, n_classes, emb=64, hidden=192, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_chars + 1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # признаки слова: последние состояния обоих направлений + max-pooling по буквам
        last = torch.cat([h[-2], h[-1]], dim=1)
        mask = (torch.arange(out.size(1))[None, :] < lengths[:, None]).unsqueeze(-1)
        pooled = out.masked_fill(~mask, -1e4).max(dim=1).values
        return self.head(torch.cat([last, pooled], dim=1))


class Encoder:
    def __init__(self, chars):
        self.idx = {c: i + 1 for i, c in enumerate(chars)}

    def batch(self, words):
        words = [w[-MAX_LEN:] for w in words]
        L = max(len(w) for w in words)
        x = torch.zeros(len(words), L, dtype=torch.long)
        for i, w in enumerate(words):
            x[i, : len(w)] = torch.tensor([self.idx.get(c, 0) for c in w])
        return x, torch.tensor([len(w) for w in words])


def any_correct_loss(logits, gold_mask):
    """Слово может иметь несколько правильных разборов (кошки = рд.ед / им.мн).
    Штрафуем модель, если СУММАРНАЯ вероятность правильных классов мала:
        loss = -log( sum_{c in gold} p(c) )"""
    all_ = torch.logsumexp(logits, dim=1)
    good = torch.logsumexp(logits.masked_fill(~gold_mask, -1e4), dim=1)
    return (all_ - good).mean()
