"""Customising Trainer by overriding `compute_loss`.

`ZLossTrainer` computes the causal-LM loss itself, adds an optional z-loss term (PaLM's penalty on
log Z = logsumexp of the logits, which keeps logits from drifting large), and logs extra metrics.
With `z_loss_weight=0` it must reproduce the default Trainer loss exactly; the tests check this.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Trainer

IGNORE_INDEX = -100


class ZLossTrainer(Trainer):
    # Our loss is divided by `num_items_in_batch` (all label tokens in the accumulated batch),
    # so Trainer must not divide it again by gradient_accumulation_steps.
    loss_is_scaled_for_ga = True

    def __init__(self, *args, z_loss_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_loss_weight = z_loss_weight
        self._metrics: dict[str, list[float]] = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")  # (B, N); keep it away from the model so it does not compute a loss
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].float()  # (B, N-1, V)
        targets = labels[:, 1:]  # (B, N-1)
        valid = targets != IGNORE_INDEX  # (B, N-1)

        if num_items_in_batch is None:  # model without **kwargs: fall back to this micro-batch only
            num_items_in_batch = valid.sum()

        ce_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
        )
        log_z = torch.logsumexp(logits, dim=-1)  # (B, N-1)
        z_sum = (log_z.pow(2) * valid).sum()
        loss = (ce_sum + self.z_loss_weight * z_sum) / num_items_in_batch

        with torch.no_grad():
            n_valid = valid.sum().clamp(min=1)
            correct = ((logits.argmax(-1) == targets) & valid).sum()
            self._metrics["token_accuracy"].append((correct / n_valid).item())
            self._metrics["mean_log_z"].append(((log_z * valid).sum() / n_valid).item())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # Average the metrics collected since the last log call, then reset them.
        for key, values in self._metrics.items():
            if values:
                logs[key] = sum(values) / len(values)
        self._metrics.clear()
        super().log(logs, start_time)
