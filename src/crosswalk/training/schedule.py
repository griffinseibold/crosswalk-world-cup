"""Learning-rate schedule: linear warmup, then cosine decay.

Implemented as a picklable callable (not a lambda) so ``LambdaLR.state_dict``
captures it and a resumed run continues on exactly the same curve.
"""

from __future__ import annotations

import math


class WarmupCosine:
    def __init__(self, total_steps: int, warmup_steps: int, min_ratio: float):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(warmup_steps, self.total_steps)
        self.min_ratio = min_ratio

    def __call__(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            # Start at 1/warmup rather than 0 so the very first step moves.
            return (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / span)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_ratio + (1 - self.min_ratio) * cosine
