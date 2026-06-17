"""Physics models for the car. SI units throughout (meters, m/s, radians, seconds).

The physics step is a pure function behind the :class:`~f1rl.physics.base.PhysicsModel`
interface; the kinematic bicycle model is the Phase 1 implementation, with a dynamic
model swapped in later behind the same contract.
"""

from f1rl.physics.base import CarState, PhysicsModel
from f1rl.physics.dynamic import DynamicBicycle, DynamicParams
from f1rl.physics.factory import make_physics
from f1rl.physics.kinematic import KinematicBicycle, KinematicParams

__all__ = [
    "CarState",
    "PhysicsModel",
    "KinematicBicycle",
    "KinematicParams",
    "DynamicBicycle",
    "DynamicParams",
    "make_physics",
]
