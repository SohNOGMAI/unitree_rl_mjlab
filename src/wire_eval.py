"""Trial-level logging and paper-table aggregation for wire traversal tests.

This module deliberately contains no controller decisions.  It observes the
existing :class:`scripts.play.WireAssistWrapper`, applies an explicit evaluation
protocol, and writes measurements that can be audited independently of the
controller's console log.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODE_TO_PRIMITIVE = {
  "ascend": "hoist",
  "step_ascent": "hoist",
  "descend": "brake",
  "step_descent": "brake",
  "gap": "swing",
  "gap_traversal": "swing",
}

TIMESERIES_FIELDS = [
  "time_s",
  "primitive",
  "condition",
  "seed",
  "wire_length_m",
  "target_wire_length_m",
  "wire_tension_N",
  "rho_z",
  "torso_roll_deg",
  "torso_pitch_deg",
  "torso_yaw_deg",
  "torso_yaw_rate_deg_s",
  "base_position_x",
  "base_position_y",
  "base_position_z",
  "back_attachment_position_x",
  "back_attachment_position_y",
  "back_attachment_position_z",
  "state_machine_state",
  "foot_ground_contact",
  "target_region_reached",
  "stable_duration_s",
]

SUMMARY_FIELDS = [
  "primitive",
  "condition",
  "seed",
  "success",
  "failure_reason",
  "completion_time_s",
  "peak_rho_z",
  "max_att_deg",
  "max_roll_deg",
  "max_pitch_deg",
  "max_yaw_rate_deg_s",
  "peak_tension_N",
  "traversal_distance_or_height",
  "final_position_error",
  "target_reached_time_s",
]

TABLE_FIELDS = [
  "primitive",
  "nominal_success",
  "assisted_success",
  "nominal_success_count",
  "nominal_trial_count",
  "assisted_success_count",
  "assisted_trial_count",
  "assisted_time_mean_s",
  "assisted_time_std_s",
  "assisted_peak_rho_z_mean",
  "assisted_peak_rho_z_std",
  "assisted_max_att_mean_deg",
  "assisted_max_att_std_deg",
]


def _float(value: Any, default: float = math.nan) -> float:
  if value is None:
    return default
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _as_numpy(value: Any) -> np.ndarray:
  if hasattr(value, "detach"):
    value = value.detach().cpu().numpy()
  return np.asarray(value)


def _quat_wxyz_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
  """Convert a normalized MuJoCo wxyz quaternion to xyz Euler angles."""
  w, x, y, z = (float(v) for v in quat)
  norm = math.sqrt(w * w + x * x + y * y + z * z)
  if norm <= 1.0e-12:
    return math.nan, math.nan, math.nan
  w, x, y, z = w / norm, x / norm, y / norm, z / norm
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  pitch = math.asin(sin_pitch)
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return roll, pitch, yaw


def _rotate_vector_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
  """Rotate a vector without adding a SciPy dependency to the logger."""
  w, x, y, z = (float(v) for v in quat)
  qvec = np.array([x, y, z], dtype=np.float64)
  v = np.asarray(vector, dtype=np.float64)
  return v + 2.0 * np.cross(qvec, np.cross(qvec, v) + w * v)


class WireTrialEvaluator:
  """Stream one trial to CSV and evaluate success without changing control."""

  def __init__(
    self,
    wire_env: Any,
    *,
    condition: str,
    seed: int,
    output_csv: str | Path,
    timeout_s: float = 20.0,
    max_attitude_deg: float = 45.0,
    stability_duration_s: float = 1.0,
  ) -> None:
    if condition not in ("nominal", "assisted"):
      raise ValueError(f"condition must be nominal or assisted, got {condition!r}")
    if wire_env.traversal_mode not in MODE_TO_PRIMITIVE:
      raise ValueError(f"unknown traversal mode: {wire_env.traversal_mode!r}")
    if timeout_s <= 0.0 or stability_duration_s <= 0.0:
      raise ValueError("timeout and stability duration must be positive")
    if not 0.0 < max_attitude_deg <= 180.0:
      raise ValueError("max attitude must be in the interval (0, 180]")

    self.wire_env = wire_env
    self.primitive = MODE_TO_PRIMITIVE[wire_env.traversal_mode]
    self.condition = condition
    self.seed = int(seed)
    self.timeout_s = float(timeout_s)
    self.max_attitude_deg = float(max_attitude_deg)
    self.stability_duration_s = float(stability_duration_s)
    self.dt = float(wire_env.dt)

    self.mass_kg = float(
      wire_env.native_model.body_subtreemass[wire_env.robot_root_id]
    )
    gravity_vector = np.asarray(wire_env.native_model.opt.gravity, dtype=np.float64)
    self.gravity_m_s2 = float(np.linalg.norm(gravity_vector))
    if self.gravity_m_s2 <= 1.0e-9:
      self.gravity_m_s2 = 9.81

    self.box_front, self.box_back, self.box_bottom, self.box_top = (
      float(v) for v in wire_env._obstacle_bounds()
    )
    initial = self._measure()
    self.initial_base = initial["base_position"].copy()
    initial_support = self.box_top if self.primitive in ("brake", "swing") else self.box_bottom
    self.initial_clearance = max(
      float(self.initial_base[2] - initial_support),
      0.45,
    )

    self.elapsed_s = 0.0
    self.previous_yaw = None
    self.stable_duration_s = 0.0
    self.target_reached_time_s = math.nan
    self.success = False
    self.failure_reason = ""
    self.finished = False
    self.max_roll_deg = 0.0
    self.max_pitch_deg = 0.0
    self.max_yaw_rate_deg_s = 0.0
    self.peak_tension_N = 0.0
    self.peak_rho_z = 0.0
    self.max_base_x = float(self.initial_base[0])
    self.max_base_z = float(self.initial_base[2])
    self.min_base_z = float(self.initial_base[2])
    self.last_measurement = initial

    self.output_csv = Path(output_csv)
    self.output_csv.parent.mkdir(parents=True, exist_ok=True)
    self._stream = self.output_csv.open("w", newline="", encoding="utf-8")
    self._writer = csv.DictWriter(self._stream, fieldnames=TIMESERIES_FIELDS)
    self._writer.writeheader()
    self._row_count = 0
    self.observe()

  def _env0(self, value: Any, item_id: int) -> np.ndarray:
    array = _as_numpy(value)
    selected = array[0, item_id] if array.ndim == 3 else array[item_id]
    return np.array(selected, dtype=np.float64, copy=True)

  def _measure(self) -> dict[str, Any]:
    data = self.wire_env.mj_data
    torso_pos = self._env0(data.xpos, self.wire_env.body_id)
    torso_quat = self._env0(data.xquat, self.wire_env.body_id)
    base_pos = self._env0(data.xpos, self.wire_env.robot_root_id)
    attachment = torso_pos + _rotate_vector_wxyz(
      torso_quat,
      self.wire_env.attachment_pos_b,
    )
    roll, pitch, yaw = _quat_wxyz_to_rpy(torso_quat)
    cable = self.wire_env.anchor_pos - attachment
    wire_length = float(np.linalg.norm(cable))
    u_z = float(cable[2] / wire_length) if wire_length > 1.0e-9 else math.nan
    tension = 0.0 if self.condition == "nominal" else float(self.wire_env.tension)
    rho_z = (
      tension * u_z / (self.mass_kg * self.gravity_m_s2)
      if math.isfinite(u_z) and self.mass_kg > 0.0
      else math.nan
    )
    target_length = (
      math.nan
      if self.condition == "nominal"
      else _float(self.wire_env.commanded_wire_length)
    )
    state = (
      "NOMINAL"
      if self.condition == "nominal"
      else self.wire_env.PHASE_NAMES.get(self.wire_env.phase, "UNKNOWN")
    )
    foot_ground_contact = self._foot_ground_contact()
    return {
      "base_position": base_pos,
      "attachment_position": attachment,
      "roll_rad": roll,
      "pitch_rad": pitch,
      "yaw_rad": yaw,
      "wire_length_m": wire_length,
      "target_wire_length_m": target_length,
      "tension_N": tension,
      "rho_z": rho_z,
      "state": state,
      "foot_ground_contact": foot_ground_contact,
    }

  def _foot_ground_contact(self) -> bool | None:
    """Return actual terrain contact from the task's existing foot sensor.

    ``None`` is reserved for configurations that do not provide the sensor;
    it is not silently interpreted as contact.
    """
    try:
      sensor = self.wire_env.env.unwrapped.scene["feet_ground_contact"]
      found = sensor.data.found
      if found is None:
        return None
      values = _as_numpy(found)
      values = values[0] if values.ndim >= 2 else values
      return bool(np.any(values > 0))
    except (AttributeError, KeyError, IndexError, TypeError):
      return None

  @staticmethod
  def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

  def _target_region_reached(self, base: np.ndarray) -> bool:
    clearance_threshold = max(0.40, 0.55 * self.initial_clearance)
    if self.primitive == "hoist":
      return bool(
        base[0] >= self.box_front + 0.20
        and base[2] >= self.box_top + clearance_threshold
      )
    if self.primitive == "brake":
      return bool(
        base[0] >= self.box_back + 0.10
        and base[2] <= self.box_bottom + self.initial_clearance + 0.20
      )
    return bool(
      base[0] >= self.box_back + 0.10
      and base[2] >= self.box_top + clearance_threshold
    )

  def _gap_contact(self, measurement: dict[str, Any]) -> bool:
    if self.primitive != "swing":
      return False
    base = measurement["base_position"]
    # MJWarp does not expose a stable, named contact-pair API here.  Detect the
    # unambiguous gross failure instead: the floating base/torso has entered
    # the gap below the two platform top surfaces.  Limb-edge contacts are not
    # inferred and are documented as unavailable in the generated README.
    return bool(
      self.box_front < base[0] < self.box_back
      and base[2] < self.box_top + 0.15
    )

  def observe(self) -> bool:
    """Record the current post-step state and return whether the trial ended."""
    if self.finished:
      return True

    measurement = self._measure()
    self.last_measurement = measurement
    base = measurement["base_position"]
    attachment = measurement["attachment_position"]
    roll_deg = math.degrees(measurement["roll_rad"])
    pitch_deg = math.degrees(measurement["pitch_rad"])
    yaw_deg = math.degrees(measurement["yaw_rad"])
    if self.previous_yaw is None:
      yaw_rate_deg_s = 0.0
    else:
      yaw_rate_deg_s = math.degrees(
        self._wrap_to_pi(measurement["yaw_rad"] - self.previous_yaw) / self.dt
      )
    self.previous_yaw = measurement["yaw_rad"]

    finite_values = np.array(
      [
        *base,
        *attachment,
        roll_deg,
        pitch_deg,
        yaw_deg,
        yaw_rate_deg_s,
        measurement["wire_length_m"],
        measurement["tension_N"],
        measurement["rho_z"],
      ],
      dtype=np.float64,
    )
    valid = bool(np.isfinite(finite_values).all())
    target_reached = valid and self._target_region_reached(base)
    if target_reached and not math.isfinite(self.target_reached_time_s):
      self.target_reached_time_s = self.elapsed_s

    abs_roll = abs(roll_deg) if math.isfinite(roll_deg) else math.inf
    abs_pitch = abs(pitch_deg) if math.isfinite(pitch_deg) else math.inf
    attitude = max(abs_roll, abs_pitch)
    if valid:
      self.max_roll_deg = max(self.max_roll_deg, abs_roll)
      self.max_pitch_deg = max(self.max_pitch_deg, abs_pitch)
      self.max_yaw_rate_deg_s = max(
        self.max_yaw_rate_deg_s,
        abs(yaw_rate_deg_s),
      )
      self.peak_tension_N = max(
        self.peak_tension_N,
        measurement["tension_N"],
      )
      self.peak_rho_z = max(self.peak_rho_z, measurement["rho_z"])
      self.max_base_x = max(self.max_base_x, float(base[0]))
      self.max_base_z = max(self.max_base_z, float(base[2]))
      self.min_base_z = min(self.min_base_z, float(base[2]))

    stable_now = bool(
      target_reached
      and measurement["foot_ground_contact"] is True
      and attitude <= self.max_attitude_deg
      and base[2] >= self.box_bottom + 0.30
    )
    self.stable_duration_s = (
      self.stable_duration_s + self.dt if stable_now else 0.0
    )

    row = {
      "time_s": self.elapsed_s,
      "primitive": self.primitive,
      "condition": self.condition,
      "seed": self.seed,
      "wire_length_m": measurement["wire_length_m"],
      "target_wire_length_m": measurement["target_wire_length_m"],
      "wire_tension_N": measurement["tension_N"],
      "rho_z": 0.0 if self.condition == "nominal" else measurement["rho_z"],
      "torso_roll_deg": roll_deg,
      "torso_pitch_deg": pitch_deg,
      "torso_yaw_deg": yaw_deg,
      "torso_yaw_rate_deg_s": yaw_rate_deg_s,
      "base_position_x": base[0],
      "base_position_y": base[1],
      "base_position_z": base[2],
      "back_attachment_position_x": attachment[0],
      "back_attachment_position_y": attachment[1],
      "back_attachment_position_z": attachment[2],
      "state_machine_state": measurement["state"],
      "foot_ground_contact": (
        "unknown"
        if measurement["foot_ground_contact"] is None
        else str(measurement["foot_ground_contact"]).lower()
      ),
      "target_region_reached": str(bool(target_reached)).lower(),
      "stable_duration_s": self.stable_duration_s,
    }
    self._writer.writerow(row)
    self._row_count += 1
    if self._row_count % 50 == 0:
      self._stream.flush()

    if not valid:
      self.failure_reason = "invalid_simulation"
      self.finished = True
    elif self._gap_contact(measurement):
      self.failure_reason = "gap_contact"
      self.finished = True
    elif attitude > 70.0 or base[2] < self.box_bottom + 0.30:
      self.failure_reason = "fall"
      self.finished = True
    elif attitude > self.max_attitude_deg:
      self.failure_reason = "excessive_attitude"
      self.finished = True
    elif self.stable_duration_s + 1.0e-9 >= self.stability_duration_s:
      self.success = True
      self.finished = True
    elif self.elapsed_s + 1.0e-9 >= self.timeout_s:
      self.failure_reason = (
        "timeout" if math.isfinite(self.target_reached_time_s)
        else "target_not_reached"
      )
      self.finished = True

    return self.finished

  def advance_time(self) -> None:
    self.elapsed_s += self.dt

  def _final_position_error(self) -> float:
    base = self.last_measurement["base_position"]
    clearance_threshold = max(0.40, 0.55 * self.initial_clearance)
    if self.primitive == "hoist":
      dx = max(self.box_front + 0.20 - base[0], 0.0)
      dz = max(self.box_top + clearance_threshold - base[2], 0.0)
    elif self.primitive == "brake":
      dx = max(self.box_back + 0.10 - base[0], 0.0)
      dz = max(base[2] - (self.box_bottom + self.initial_clearance + 0.20), 0.0)
    else:
      dx = max(self.box_back + 0.10 - base[0], 0.0)
      dz = max(self.box_top + clearance_threshold - base[2], 0.0)
    return float(math.hypot(dx, dz))

  def finalize(self) -> dict[str, Any]:
    if not self.finished:
      self.failure_reason = (
        "timeout" if math.isfinite(self.target_reached_time_s)
        else "target_not_reached"
      )
      self.finished = True
    self.close()
    if self.primitive == "hoist":
      traversal = self.max_base_z - float(self.initial_base[2])
    elif self.primitive == "brake":
      traversal = float(self.initial_base[2]) - self.min_base_z
    else:
      traversal = self.max_base_x - float(self.initial_base[0])
    return {
      "primitive": self.primitive,
      "condition": self.condition,
      "seed": self.seed,
      "success": str(self.success).lower(),
      "failure_reason": "" if self.success else self.failure_reason,
      # Completion includes the requested post-traversal stability window.
      "completion_time_s": self.elapsed_s,
      "peak_rho_z": 0.0 if self.condition == "nominal" else self.peak_rho_z,
      "max_att_deg": max(self.max_roll_deg, self.max_pitch_deg),
      "max_roll_deg": self.max_roll_deg,
      "max_pitch_deg": self.max_pitch_deg,
      "max_yaw_rate_deg_s": self.max_yaw_rate_deg_s,
      "peak_tension_N": self.peak_tension_N,
      "traversal_distance_or_height": traversal,
      "final_position_error": self._final_position_error(),
      "target_reached_time_s": self.target_reached_time_s,
    }

  def close(self) -> None:
    """Flush a partial CSV safely if simulation raises between samples."""
    if not self._stream.closed:
      self._stream.flush()
      self._stream.close()


def invalid_trial_summary(
  primitive: str,
  condition: str,
  seed: int,
  output_csv: str | Path,
) -> dict[str, Any]:
  """Create a schema-valid CSV/summary when environment construction fails."""
  path = Path(output_csv)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=TIMESERIES_FIELDS)
    writer.writeheader()
  row = {field: math.nan for field in SUMMARY_FIELDS}
  row.update(
    primitive=primitive,
    condition=condition,
    seed=seed,
    success="false",
    failure_reason="invalid_simulation",
    peak_rho_z=0.0 if condition == "nominal" else math.nan,
  )
  return row


def write_summary_trials(
  summaries: Iterable[dict[str, Any]],
  path: str | Path,
) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for summary in summaries:
      writer.writerow(summary)


def _is_success(row: dict[str, Any]) -> bool:
  return str(row.get("success", "")).strip().lower() in ("true", "1", "yes")


def _mean_std(values: Iterable[Any]) -> tuple[float, float]:
  finite = np.asarray(
    [value for raw in values if math.isfinite(value := _float(raw))],
    dtype=np.float64,
  )
  if finite.size == 0:
    return math.nan, math.nan
  mean = float(np.mean(finite))
  std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
  return mean, std


def build_summary_table(
  summaries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
  rows = list(summaries)
  table: list[dict[str, Any]] = []
  for primitive in ("hoist", "brake", "swing"):
    nominal = [
      row for row in rows
      if row.get("primitive") == primitive and row.get("condition") == "nominal"
    ]
    assisted = [
      row for row in rows
      if row.get("primitive") == primitive and row.get("condition") == "assisted"
    ]
    nominal_successes = sum(_is_success(row) for row in nominal)
    assisted_success_rows = [row for row in assisted if _is_success(row)]
    assisted_successes = len(assisted_success_rows)
    time_mean, time_std = _mean_std(
      row.get("completion_time_s") for row in assisted_success_rows
    )
    rho_mean, rho_std = _mean_std(
      row.get("peak_rho_z") for row in assisted
    )
    att_mean, att_std = _mean_std(
      row.get("max_att_deg") for row in assisted
    )
    table.append(
      {
        "primitive": primitive,
        "nominal_success": f"{nominal_successes}/{len(nominal)}",
        "assisted_success": f"{assisted_successes}/{len(assisted)}",
        "nominal_success_count": nominal_successes,
        "nominal_trial_count": len(nominal),
        "assisted_success_count": assisted_successes,
        "assisted_trial_count": len(assisted),
        "assisted_time_mean_s": time_mean,
        "assisted_time_std_s": time_std,
        "assisted_peak_rho_z_mean": rho_mean,
        "assisted_peak_rho_z_std": rho_std,
        "assisted_max_att_mean_deg": att_mean,
        "assisted_max_att_std_deg": att_std,
      }
    )
  return table


def write_summary_table(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def load_summary_trials(path: str | Path) -> list[dict[str, str]]:
  with Path(path).open(newline="", encoding="utf-8") as stream:
    return list(csv.DictReader(stream))
