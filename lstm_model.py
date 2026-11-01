"""
Next-icon LSTM for the AAC board.

A real (small) recurrent neural network: embedding -> single LSTM layer ->
softmax over the icon vocabulary. Trained with backpropagation through time
and Adam. Written in plain numpy so the server only needs numpy at runtime
(no PyTorch / TensorFlow, which don't fit on Render's free tier).

Unlike a bigram, the hidden state carries the *whole* sequence typed so far,
so "sick -> teacher" and "happy -> teacher" can lead to different predictions.
"""

import numpy as np

BOS = "<bos>"   # start-of-sentence token, fed as the first input


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NextIconLSTM:
    def __init__(self, vocab: list[str], embed_dim: int = 24, hidden_dim: int = 48, seed: int = 0):
        if vocab[0] != BOS:
            vocab = [BOS] + [v for v in vocab if v != BOS]
        self.vocab = list(vocab)
        self.index = {tok: i for i, tok in enumerate(self.vocab)}
        V, D, H = len(self.vocab), embed_dim, hidden_dim
        self.D, self.H = D, H

        rng = np.random.default_rng(seed)
        self.params = {
            "E":  rng.normal(0, 0.1, (V, D)),
            "W":  rng.normal(0, 1.0 / np.sqrt(D + H), (4 * H, D + H)),   # gates: i, f, o, g
            "b":  np.zeros(4 * H),
            "Wy": rng.normal(0, 1.0 / np.sqrt(H), (V, H)),
            "by": np.zeros(V),
        }
        self.params["b"][H:2 * H] = 1.0   # forget-gate bias = 1 helps early training

    # ── Core recurrence ────────────────────────────────────────────────────────
    def _step(self, x_idx, h_prev, c_prev):
        p, H = self.params, self.H
        e = p["E"][x_idx]
        z = np.concatenate([e, h_prev])
        a = p["W"] @ z + p["b"]
        i = _sigmoid(a[:H])
        f = _sigmoid(a[H:2 * H])
        o = _sigmoid(a[2 * H:3 * H])
        g = np.tanh(a[3 * H:])
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
        return h, c, (x_idx, z, i, f, o, g, c_prev, tc)

    def encode(self, sequence: list[str]) -> list[int]:
        """BOS + known icon ids (unknown ids are dropped)."""
        return [0] + [self.index[t] for t in sequence if t in self.index and t != BOS]

    # ── Inference ──────────────────────────────────────────────────────────────
    def next_probs(self, sequence: list[str]) -> np.ndarray:
        """Probability distribution over the vocab for the next icon."""
        h, c = np.zeros(self.H), np.zeros(self.H)
        for x in self.encode(sequence):
            h, c, _ = self._step(x, h, c)
        logits = self.params["Wy"] @ h + self.params["by"]
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def predict(self, sequence: list[str], top_k: int = 3, exclude: set[str] | None = None):
        """Top-k (icon_id, probability) pairs, skipping BOS, the icon just
        tapped (an immediate repeat is never useful) and excluded ids."""
        probs = self.next_probs(sequence)
        skip = {BOS} | set(sequence[-1:]) | (exclude or set())
        ranked = [(self.vocab[i], float(probs[i])) for i in np.argsort(-probs)
                  if self.vocab[i] not in skip]
        return ranked[:top_k]

    # ── Training ───────────────────────────────────────────────────────────────
    def _loss_and_grads(self, seq_idx: list[int]):
        """Next-token cross-entropy over one sequence, plus gradients (BPTT)."""
        p, H, D = self.params, self.H, self.D
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        inputs, targets = seq_idx[:-1], seq_idx[1:]

        h, c = np.zeros(H), np.zeros(H)
        caches, hs, probs_list = [], [], []
        loss = 0.0
        for x, y in zip(inputs, targets):
            h, c, cache = self._step(x, h, c)
            logits = p["Wy"] @ h + p["by"]
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            loss -= np.log(probs[y] + 1e-12)
            caches.append(cache)
            hs.append(h)
            probs_list.append(probs)

        dh_next, dc_next = np.zeros(H), np.zeros(H)
        for t in reversed(range(len(inputs))):
            x_idx, z, i, f, o, g, c_prev, tc = caches[t]
            dlogits = probs_list[t].copy()
            dlogits[targets[t]] -= 1.0
            grads["Wy"] += np.outer(dlogits, hs[t])
            grads["by"] += dlogits

            dh = p["Wy"].T @ dlogits + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            di, dg, df = dc * g, dc * i, dc * c_prev
            dc_next = dc * f

            da = np.concatenate([
                di * i * (1 - i),
                df * f * (1 - f),
                do * o * (1 - o),
                dg * (1 - g ** 2),
            ])
            grads["W"] += np.outer(da, z)
            grads["b"] += da
            dz = p["W"].T @ da
            grads["E"][x_idx] += dz[:D]
            dh_next = dz[D:]

        return loss, len(targets), grads

    def train(self, sequences: list[list[str]], epochs: int = 40, lr: float = 0.01,
              weight_decay: float = 1e-4, clip: float = 5.0, seed: int = 0, verbose: bool = True):
        """Adam over individual sequences. Returns per-epoch average loss."""
        data = [s for s in (self.encode(seq) for seq in sequences) if len(s) >= 2]
        if not data:
            raise ValueError("No usable training sequences")

        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        history = []

        for epoch in range(epochs):
            total, count = 0.0, 0
            for idx in rng.permutation(len(data)):
                loss, n, grads = self._loss_and_grads(data[idx])
                total += loss
                count += n
                norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
                scale = min(1.0, clip / (norm + 1e-12))
                step += 1
                for k in self.params:
                    g = grads[k] * scale + weight_decay * self.params[k]
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    m_hat = m[k] / (1 - b1 ** step)
                    v_hat = v[k] / (1 - b2 ** step)
                    self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            history.append(total / count)
            if verbose:
                print(f"epoch {epoch + 1:3d}/{epochs}  loss {history[-1]:.4f}")
        return history

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str):
        np.savez_compressed(path, vocab=np.array(self.vocab), **self.params)

    @classmethod
    def load(cls, path: str) -> "NextIconLSTM":
        with np.load(path, allow_pickle=False) as data:
            vocab = [str(t) for t in data["vocab"]]
            model = cls(vocab, embed_dim=data["E"].shape[1], hidden_dim=data["Wy"].shape[1])
            for k in model.params:
                model.params[k] = data[k].astype(np.float64)
        return model
