"""Pure, itemized state-cost calculations shared by collection paths."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GuiCosts:
    run: int
    potential: object
    potential_delta: float
    dwell_cost: float

    @property
    def total(self):
        return self.potential_delta + self.dwell_cost


def gui_costs(opened, run=0, previous=None, *, weight=0.0,
              dwell_steps=200.0, step_cost=0.0, grace_steps=40):
    if (not all(math.isfinite(float(x)) for x in (weight, dwell_steps, step_cost))
            or weight < 0 or step_cost < 0 or grace_steps < 0):
        raise ValueError('GUI costs must be finite and nonnegative')
    if weight <= 0 and step_cost <= 0:
        return GuiCosts(run, previous, 0.0, 0.0)
    run = run + 1 if opened else 0
    phi = -min(1.0, run/max(1.0, dwell_steps)) if weight > 0 else previous
    delta = weight*(phi-previous) if weight > 0 and previous is not None else 0.0
    cost = -step_cost if opened and run > grace_steps else 0.0
    return GuiCosts(run, phi, delta, cost)
