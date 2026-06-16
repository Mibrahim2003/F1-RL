"""Car state and the physics-model interface (TECHNICAL_DESIGN.md §5).

The state carries the full struct used by later phases; Phase 1's kinematic model
only reads and writes the kinematic subset (``x, y, yaw`` and a scalar speed derived
from ``vx``). Keeping the full struct now means the dynamic model swaps in later with
no change to the env or server contract.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Protocol


@dataclass
class CarState:
    """Pose and motion of a single car, in SI units, world frame.

    The world frame is right-handed with ``x`` right and ``y`` up; ``yaw`` is the
    heading in radians in the standard mathematical sense (0 along +x, CCW positive).
    """

    x: float = 0.0  # meters, world frame
    y: float = 0.0  # meters, world frame
    yaw: float = 0.0  # radians
    vx: float = 0.0  # m/s, longitudinal in body frame
    vy: float = 0.0  # m/s, lateral in body frame (0 for the kinematic model)
    yaw_rate: float = 0.0  # rad/s
    tire_wear: float = 0.0  # 0..1 (unused in Phase 1)
    compound: int = 0  # 0 soft, 1 medium, 2 hard, 3 intermediate, 4 wet

    @property
    def speed(self) -> float:
        """Scalar ground speed (m/s). For the kinematic model this is ``|vx|``."""
        return math.hypot(self.vx, self.vy)

    def copy(self) -> CarState:
        """Return a shallow copy (all fields are immutable scalars)."""
        return dataclasses.replace(self)


class PhysicsModel(Protocol):
    """A car physics model.

    The step is a **pure function** of state, controls, grip, and dt: no globals, no
    rendering, no track lookups. ``grip`` is the single scalar from the grip pipeline
    (1.0 in Phase 1); the dynamic model uses it to size the friction circle.
    """

    def step(
        self,
        state: CarState,
        steer: float,
        longitudinal: float,
        grip: float,
        dt: float,
    ) -> CarState:
        """Advance ``state`` by ``dt`` seconds.

        Args:
            state: Current car state (not mutated).
            steer: Steering command in ``[-1, 1]`` (maps to ``[-max_steer, +max_steer]``).
            longitudinal: Throttle/brake command in ``[-1, 1]`` (>=0 throttle, <0 brake).
            grip: Grip scalar gating tire force (1.0 for the kinematic model).
            dt: Integration timestep in seconds.

        Returns:
            The new :class:`CarState`.
        """
        ...
