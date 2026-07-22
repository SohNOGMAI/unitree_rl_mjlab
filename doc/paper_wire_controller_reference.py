"""Minimal reference implementation for the wire-assist control law.

This file is a paper-oriented extraction, not a replacement for scripts/play.py.
It makes the geometry planner, bounded cable-length trajectory, unilateral
tension controller, and tension slew-rate limiter explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


Array = np.ndarray


class Mode(Enum):
  ASCEND = auto()
  DESCEND = auto()
  GAP = auto()


class Phase(Enum):
  APPROACH = auto()
  SETTLE = auto()
  LIFT = auto()
  HOLD = auto()
  LOWER = auto()
  ASSIST = auto()
  RELEASE = auto()
  DONE = auto()


@dataclass(frozen=True)
class CableProfile:
  """Three scalar cable lengths generated from geometric waypoints."""

  start: float
  hoist: float
  release: float

  @property
  def reel_in(self) -> float:
    return max(self.start - self.hoist, 0.0)

  @property
  def payout(self) -> float:
    return max(self.release - self.hoist, 0.0)


@dataclass(frozen=True)
class WinchLimits:
  reel_speed: float
  payout_speed: float
  reel_braking_accel: float
  payout_braking_accel: float
  max_tension: float
  tension_slew_rate: float
  release_slew_rate: float


@dataclass(frozen=True)
class CableGains:
  kp: float
  kd: float
  minimum_vertical_fraction: float = 0.05


def cable_geometry(anchor_w: Array, attachment_w: Array) -> tuple[float, Array]:
  """Return cable length and unit direction from robot to anchor."""
  vector = np.asarray(anchor_w, dtype=float) - np.asarray(attachment_w, dtype=float)
  length = float(np.linalg.norm(vector))
  if length <= 1.0e-9:
    raise ValueError("anchor and attachment point must be distinct")
  return length, vector / length


def plan_profile(
  anchor_w: Array,
  attachment_start_w: Array,
  attachment_hoist_w: Array,
  attachment_release_w: Array,
  hoist_calibration: float = 0.0,
  release_calibration: float = 0.0,
) -> CableProfile:
  """Map known geometric waypoints to a common length profile.

  Calibration terms represent measured cable-routing or compliance bias.  They
  must be reported as experimental parameters rather than hidden offsets.
  """
  l_start, _ = cable_geometry(anchor_w, attachment_start_w)
  l_hoist, _ = cable_geometry(anchor_w, attachment_hoist_w)
  l_release, _ = cable_geometry(anchor_w, attachment_release_w)
  l_hoist = max(l_hoist + hoist_calibration, 0.05)
  l_release = max(l_release + release_calibration, l_hoist)
  return CableProfile(l_start, l_hoist, l_release)


def approach_target(
  current: float,
  target: float,
  speed_limit: float,
  braking_accel: float,
  dt: float,
) -> tuple[float, float]:
  """Move a scalar length toward its target with a stopping-distance profile."""
  distance = abs(target - current)
  if distance <= 1.0e-12:
    return target, 0.0
  speed = min(speed_limit, np.sqrt(2.0 * braking_accel * distance))
  signed_speed = np.copysign(speed, target - current)
  next_value = current + signed_speed * dt
  if (target - current) * (target - next_value) <= 0.0:
    return target, 0.0
  return float(next_value), float(signed_speed)


def desired_tension(
  mode: Mode,
  phase: Phase,
  robot_mass: float,
  gravity: float,
  cable_direction: Array,
  actual_length: float,
  actual_rate: float,
  commanded_length: float,
  commanded_rate: float,
  gains: CableGains,
  tangential_speed: float = 0.0,
) -> float:
  """Compute unilateral tension from signed length and rate errors.

  Vertical hoisting uses a quasi-static vertical support term mg/u_z.  Gap
  swing uses the radial pendulum term mg*u_z + m*v_t^2/L.  The production code
  currently omits the explicit centripetal term and lets feedback supply it.
  """
  u_z = max(float(cable_direction[2]), gains.minimum_vertical_fraction)
  length_error = actual_length - commanded_length
  rate_error = actual_rate - commanded_rate

  if mode is Mode.GAP and phase in (Phase.LIFT, Phase.HOLD):
    feedforward = (
      robot_mass * gravity * u_z
      + robot_mass * tangential_speed**2 / max(actual_length, 0.05)
    )
  else:
    feedforward = robot_mass * gravity / u_z

  raw = feedforward + gains.kp * length_error + gains.kd * rate_error
  return max(float(raw), 0.0)


def limit_tension(
  previous: float,
  desired: float,
  phase: Phase,
  limits: WinchLimits,
  dt: float,
) -> float:
  """Apply magnitude and phase-dependent slew-rate constraints."""
  target = float(np.clip(desired, 0.0, limits.max_tension))
  slew = (
    limits.release_slew_rate
    if phase is Phase.RELEASE and target < previous
    else limits.tension_slew_rate
  )
  return previous + float(np.clip(target - previous, -slew * dt, slew * dt))


def attachment_wrench(
  attachment_w: Array,
  body_com_w: Array,
  cable_direction: Array,
  tension: float,
) -> Array:
  """Return [force, torque] applied at the torso COM."""
  force = float(tension) * np.asarray(cable_direction, dtype=float)
  moment_arm = np.asarray(attachment_w, dtype=float) - np.asarray(body_com_w, dtype=float)
  torque = np.cross(moment_arm, force)
  return np.concatenate((force, torque))


def update_length_command(
  phase: Phase,
  current_command: float,
  profile: CableProfile,
  limits: WinchLimits,
  dt: float,
) -> tuple[float, float]:
  """Update L_cmd for the active hybrid-control phase."""
  if phase is Phase.LIFT:
    return approach_target(
      current_command,
      profile.hoist,
      limits.reel_speed,
      limits.reel_braking_accel,
      dt,
    )
  if phase is Phase.LOWER:
    return approach_target(
      current_command,
      profile.release,
      limits.payout_speed,
      limits.payout_braking_accel,
      dt,
    )
  if phase in (Phase.HOLD,):
    return profile.hoist, 0.0
  if phase in (Phase.ASSIST, Phase.RELEASE):
    return profile.release, 0.0
  return current_command, 0.0

