"""foundation.geometry — typed state and geometry library (plan Stage 4).

NumPy float64 reference implementation, self-contained (no imports from the
live agent). Differentiable PyTorch versions of the quaternion and rotor
operations live in `foundation.geometry.torch_ops`, imported explicitly so the
core never requires torch. Offline/shadow only: nothing here touches live
reward, actions, replay or normalisation.

Modules:
  units        Unit, Quantity (unit + frame + scale status + std + provenance)
  values       UNKNOWN / INAPPLICABLE, DiscreteVar, RelationType/Relation, Timestamp
  transforms   Hamilton quaternions, SE3, SE2, slerp (conventions in its docstring)
  frames       FrameGraph (explicit conversion), AxisConvention (adapter metadata)
  conventions  declared conventions (minecraft, ros_flu) as data
  rotor        Cl(3,0) rotors, declared basis/sign/sandwich conventions
  constraints  Constraint, ConstraintRegistry, Diagnostic + example laws
"""

from .errors import (ClockMismatchError, ConstraintError, DegenerateError,
                     DisconnectedFramesError, DomainError, FrameError,
                     FrameMismatchError, GeometryError, ScaleStatusError,
                     UnitError, UnknownFrameError)
from .units import (ANGLE, DIMENSIONLESS, KG, LENGTH, M, MASS, RAD, S,
                    SCALE_STATUSES, TIME, Quantity, Unit, declare_unit)
from .values import (ABSENT, INAPPLICABLE, UNKNOWN, DiscreteVar, Relation, RelationType,
                     Timestamp, is_unknown)
from .transforms import (SE2, SE3, matrix_to_quat, quat_angle, quat_conj,
                         quat_equal, quat_from_axis_angle, quat_identity,
                         quat_inv, quat_mul, quat_normalize, quat_rotate,
                         quat_to_matrix, random_quat, slerp)
from .frames import AxisConvention, FrameGraph
from .conventions import (MINECRAFT, ROS_FLU, declare_convention,
                          declared_conventions, get_convention)
from .rotor import Rotor, gp, reverse
from .constraints import (Constraint, ConstraintRegistry, Diagnostic,
                          conservation_constraint, coordinate_change,
                          gravity_coordinate_covariance_constraint,
                          incline_acceleration, physical_rotation,
                          rigid_distance_constraint)

__all__ = [n for n in dir() if not n.startswith("_")]
