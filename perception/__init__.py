"""Vision analysis + relative localization for a chassis with a robot arm.

Stitched BEV + YOLO-World grasp xy + occupancy (avoidance / path reference).
"""

from perception.schema import SCHEMA_VERSION, GraspResult, NavResult, PerceptionEvent

__all__ = [
    "SCHEMA_VERSION",
    "NavResult",
    "GraspResult",
    "PerceptionEvent",
]
