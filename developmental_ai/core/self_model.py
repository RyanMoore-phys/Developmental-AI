"""Self-model of competence (Rung 5 — metacognition).

A small ONLINE model that predicts the agent's own probability of success per
task from a task descriptor, trained on the agent's actual episode outcomes.
Two things make it a genuine metacognitive signal rather than a bookkeeping
tally:

  1. CALIBRATION — it predicts P(success) BEFORE seeing the outcome, and we
     measure Expected Calibration Error (ECE): "when it says 70%, does it
     succeed ~70%?". A learned logistic model can be over/under-confident and
     lags the true competence, so calibration is a real, falsifiable property.

  2. LEARNING PROGRESS — it tracks the recent change in its own predicted
     competence per task, which is what a competence-driven curriculum uses to
     decide where practice will pay off ("knows what it doesn't know, and what
     it can learn next").

The model itself is identical across experiment arms; the arms differ only in
whether the ALLOCATION reads the true self-model, a scrambled view, or ignores
it (uniform). That isolates the causal value of an accurate self-model.
"""
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn


class CompetencePredictor(nn.Module):
    def __init__(self, n_tasks: int, lr: float = 0.05, lp_window: int = 15):
        # floor for per-slot logits — see the clamp in update()
        self.min_logit = -2.0
        super().__init__()
        self.n_tasks = int(n_tasks)
        # Online logistic self-model: one learnable logit per task (a Linear over
        # the task one-hot). Trained with BCE on each (task, success) outcome.
        self.head = nn.Linear(self.n_tasks, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # FREEZE the shared bias (audit fix): logit(task) = w[task] + b, and
        # every BCE update moved b for ALL tasks — a failure-dominated stream
        # (the lifelong norm) dragged every slot's competence down together,
        # locking the whole option set below competence_floor at once. Frozen
        # at 0, each task's logit is its own w[task]: fully decoupled. Loaded
        # legacy states keep their stored bias VALUE (a constant offset), but
        # it can no longer drift.
        self.head.bias.requires_grad_(False)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.bce = nn.BCEWithLogitsLoss()
        self._pred_hist: List[deque] = [deque(maxlen=lp_window) for _ in range(self.n_tasks)]
        self._calib: List[tuple] = []   # (prediction, outcome) pairs for ECE
        self._n_updates = [0 for _ in range(self.n_tasks)]

    def _onehot(self, task_id: int) -> torch.Tensor:
        x = torch.zeros(1, self.n_tasks)
        x[0, int(task_id)] = 1.0
        return x

    @torch.no_grad()
    def predict(self, task_id: int) -> float:
        return float(torch.sigmoid(self.head(self._onehot(task_id))).item())

    @torch.no_grad()
    def predict_all(self) -> np.ndarray:
        x = torch.eye(self.n_tasks)
        return torch.sigmoid(self.head(x)).reshape(-1).cpu().numpy()

    def update(self, task_id: int, success: float) -> None:
        """One online BCE step on a single (task, success) outcome. Records the
        pre-update prediction for calibration and learning-progress tracking."""
        pre = self.predict(task_id)                      # belief BEFORE this outcome
        self._calib.append((pre, float(success)))
        self._pred_hist[int(task_id)].append(pre)
        self._n_updates[int(task_id)] += 1
        x = self._onehot(task_id)
        y = torch.tensor([[float(success)]])
        loss = self.bce(self.head(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # ---- RECOVERABILITY CLAMP (2026-07-27 stall assessment) ----------
        # Measured live: all 10 bound slots sat at competence 0.0004-0.041
        # against a 0.25 floor, so mask() rejected EVERY learned option on
        # EVERY decision for the entire run. Slot 25 was at logit -7.78 —
        # that does not encode "incompetent", it encodes "needs ~134
        # consecutive successes at this lr to ever be offered again". A
        # probability that cannot climb back is a LATCH wearing a
        # probability's clothes (the 6th in this project; AUDIT #46 froze
        # the shared bias but never bounded the per-slot ratchet).
        #
        # Clamping at -2.0 (competence 0.119) PRESERVES the gate's
        # discrimination — still far below the 0.25 floor, still gated —
        # while making recovery reachable in ~20 successes instead of 134.
        with torch.no_grad():
            self.head.weight[0, int(task_id)].clamp_(min=self.min_logit)

    def learning_progress(self, task_id: int) -> float:
        """Recent rise in predicted competence for a task (>=0). Zero until the
        task has been practiced enough to have a history."""
        h = self._pred_hist[int(task_id)]
        if len(h) < 3:
            return 0.0
        return max(0.0, float(h[-1] - h[0]))

    def n_updates(self, task_id: int) -> int:
        return self._n_updates[int(task_id)]

    # ---- slot paging (lifelong working-set <-> long-term store) ----
    # A slot's ENTIRE competence state is one logit (head.weight[0, slot]) plus
    # its prediction history and update count — the shared bias is not
    # per-slot. Paging a working slot out/in serializes and restores exactly
    # this so a recalled goal resumes its learned competence instead of
    # re-learning from the sigmoid(0)=0.5 prior.
    @torch.no_grad()
    def export_slot(self, task_id: int) -> dict:
        i = int(task_id)
        return {"w": float(self.head.weight[0, i].item()),
                "hist": [float(v) for v in self._pred_hist[i]],
                "n": int(self._n_updates[i])}

    @torch.no_grad()
    def reset_slot(self, task_id: int) -> None:
        """Return a slot to the fresh (unlearned) prior so a new occupant does
        not inherit the evicted goal's competence."""
        i = int(task_id)
        self.head.weight[0, i] = 0.0
        self._pred_hist[i].clear()
        self._n_updates[i] = 0

    @torch.no_grad()
    def import_slot(self, task_id: int, state: dict) -> None:
        i = int(task_id)
        self.head.weight[0, i] = float((state or {}).get("w", 0.0))
        self._pred_hist[i].clear()
        for v in (state or {}).get("hist", []):
            self._pred_hist[i].append(float(v))
        self._n_updates[i] = int((state or {}).get("n", 0))

    # ---- calibration ----
    def ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error over all recorded (pred, outcome) pairs."""
        if not self._calib:
            return float("nan")
        preds = np.array([p for p, _ in self._calib], dtype=np.float64)
        outs = np.array([o for _, o in self._calib], dtype=np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(preds)
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            m = (preds > lo) & (preds <= hi) if b > 0 else (preds >= lo) & (preds <= hi)
            if not m.any():
                continue
            conf = preds[m].mean()
            acc = outs[m].mean()
            ece += (m.sum() / n) * abs(conf - acc)
        return float(ece)

    def reliability(self, n_bins: int = 10):
        """(bin_centers, confidence, accuracy, count) for a reliability diagram."""
        preds = np.array([p for p, _ in self._calib], dtype=np.float64)
        outs = np.array([o for _, o in self._calib], dtype=np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            m = (preds > lo) & (preds <= hi) if b > 0 else (preds >= lo) & (preds <= hi)
            if not m.any():
                continue
            rows.append(((lo + hi) / 2, float(preds[m].mean()),
                         float(outs[m].mean()), int(m.sum())))
        return rows

    def brier(self) -> float:
        if not self._calib:
            return float("nan")
        preds = np.array([p for p, _ in self._calib], dtype=np.float64)
        outs = np.array([o for _, o in self._calib], dtype=np.float64)
        return float(((preds - outs) ** 2).mean())
