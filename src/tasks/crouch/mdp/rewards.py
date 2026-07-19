"""Reward terms for a planted-feet crouch transition."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT_CFG = SceneEntityCfg("robot")


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward every non-terminal step so long-horizon balance dominates."""
  return (~env.termination_manager.terminated).float()


def _body_roll_pitch(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return world-referenced roll and pitch for one selected body.

  Projected gravity is yaw invariant, which is important because crouch resets
  retain a small randomized heading.  Positive pitch means leaning toward the
  robot's +x (forward) direction.
  """
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  if body_quat_w.shape[1] != 1:
    raise ValueError("Body attitude terms require exactly one selected body")
  projected_gravity_b = quat_apply_inverse(
    body_quat_w[:, 0, :], asset.data.gravity_vec_w
  )
  roll = torch.atan2(
    -projected_gravity_b[:, 1], -projected_gravity_b[:, 2]
  )
  pitch = torch.atan2(
    projected_gravity_b[:, 0], -projected_gravity_b[:, 2]
  )
  return roll, pitch


class target_joint_posture_error:
  """Squared physical joint error to the command-interpolated pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.target = asset.data.default_joint_pos.clone()
    joint_names = asset.joint_names
    self.left_hip_roll_id = joint_names.index("left_hip_roll_joint")
    self.right_hip_roll_id = joint_names.index("right_hip_roll_joint")
    for pattern, value in cfg.params["target_positions"].items():
      matched = False
      for joint_id, name in enumerate(joint_names):
        if re.fullmatch(pattern, name):
          self.target[:, joint_id] = float(value)
          matched = True
      if not matched:
        raise ValueError(f"Crouch target pattern matched no joint: {pattern!r}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_positions: dict[str, float],
    command_name: str,
    command_index: int,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del target_positions
    asset: Entity = env.scene[asset_cfg.name]
    crouch_depth = torch.clamp(
      env.command_manager.get_command(command_name)[:, command_index],
      min=0.0,
      max=1.0,
    )
    target = asset.data.default_joint_pos + crouch_depth.unsqueeze(1) * (
      self.target - asset.data.default_joint_pos
    )
    error = asset.data.joint_pos - target
    # Only score the joints selected by SceneEntityCfg.  The remaining joints
    # are free to provide the active balance compensation that a statically
    # prescribed whole-body pose cannot express.
    error = error[:, asset_cfg.joint_ids]
    rms = torch.sqrt(torch.mean(torch.square(error), dim=1))
    env.extras["log"]["Metrics/crouch_joint_rms"] = torch.mean(rms)
    env.extras["log"]["Metrics/crouch_command"] = torch.mean(crouch_depth)
    env.extras["log"]["Metrics/crouch_left_hip_roll_deg"] = torch.rad2deg(
      torch.mean(asset.data.joint_pos[:, self.left_hip_roll_id])
    )
    env.extras["log"]["Metrics/crouch_right_hip_roll_deg"] = torch.rad2deg(
      torch.mean(asset.data.joint_pos[:, self.right_hip_roll_id])
    )
    env.extras["log"]["Metrics/crouch_target_hip_roll_deg"] = torch.rad2deg(
      torch.mean(target[:, self.left_hip_roll_id])
    )
    return torch.mean(torch.square(error), dim=1)


class target_joint_action_error:
  """Squared normalized-action error to the commanded crouch pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    action_term = env.action_manager.get_term(cfg.params["action_name"])
    self.target = torch.zeros_like(action_term.raw_action)
    scale = action_term.scale
    if not isinstance(scale, torch.Tensor):
      scale = torch.full_like(self.target, float(scale))
    elif scale.ndim == 1:
      scale = scale.unsqueeze(0)
    for pattern, value in cfg.params["target_positions"].items():
      matched = False
      for action_id, name in enumerate(action_term.target_names):
        if not re.fullmatch(pattern, name):
          continue
        joint_id = asset.joint_names.index(name)
        default = asset.data.default_joint_pos[:, joint_id]
        self.target[:, action_id] = (
          float(value) - default
        ) / scale[:, action_id]
        matched = True
      if not matched:
        raise ValueError(f"Crouch action target matched no joint: {pattern!r}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_positions: dict[str, float],
    command_name: str,
    command_index: int,
    action_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del target_positions, asset_cfg
    depth = torch.clamp(
      env.command_manager.get_command(command_name)[:, command_index],
      min=0.0,
      max=1.0,
    )
    action = env.action_manager.get_term(action_name).raw_action
    desired = depth.unsqueeze(1) * self.target
    rms = torch.sqrt(torch.mean(torch.square(action - desired), dim=1))
    env.extras["log"]["Metrics/crouch_action_rms"] = torch.mean(rms)
    return torch.square(rms)


class target_joint_posture(target_joint_posture_error):
  """Backward-compatible exponential tracker used by the 23-DoF prototype."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_positions: dict[str, float],
    std: float,
    command_name: str,
    command_index: int,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    error_sq = super().__call__(
      env,
      target_positions,
      command_name,
      command_index,
      asset_cfg,
    )
    return torch.exp(-error_sq / std**2)


class target_joint_action(target_joint_action_error):
  """Backward-compatible exponential tracker used by the 23-DoF prototype."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_positions: dict[str, float],
    std: float,
    command_name: str,
    command_index: int,
    action_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    error_sq = super().__call__(
      env,
      target_positions,
      command_name,
      command_index,
      action_name,
      asset_cfg,
    )
    return torch.exp(-error_sq / std**2)


def target_root_height_error(
  env: ManagerBasedRlEnv,
  start_height: float,
  target_height: float,
  command_name: str,
  command_index: int,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Track a command-conditioned pelvis height from standing to crouching."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  crouch_depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, command_index],
    min=0.0,
    max=1.0,
  )
  commanded_height = start_height + crouch_depth * (target_height - start_height)
  error = height - commanded_height
  env.extras["log"]["Metrics/crouch_root_height"] = torch.mean(height)
  return torch.square(error)


def target_body_attitude_error(
  env: ManagerBasedRlEnv,
  target_pitch_at_full_depth: float,
  roll_scale: float,
  command_name: str,
  command_index: int,
  metric_prefix: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track a forward body lean while keeping lateral attitude level.

  The desired pitch is interpolated with the same normalized depth command as
  the joint reference.  Standing therefore retains a zero-pitch target, while
  the torso and pelvis lean forward gradually during the crouch.
  """
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch = _body_roll_pitch(asset, asset_cfg)
  depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, command_index],
    min=0.0,
    max=1.0,
  )
  target_pitch = depth * target_pitch_at_full_depth
  pitch_error = pitch - target_pitch
  env.extras["log"][f"Metrics/{metric_prefix}_pitch_deg"] = (
    torch.rad2deg(torch.mean(pitch))
  )
  env.extras["log"][f"Metrics/{metric_prefix}_target_pitch_deg"] = (
    torch.rad2deg(torch.mean(target_pitch))
  )
  env.extras["log"][f"Metrics/{metric_prefix}_roll_deg"] = (
    torch.rad2deg(torch.mean(roll))
  )
  return torch.square(pitch_error) + roll_scale * torch.square(roll)


def body_roll_level_error(
  env: ManagerBasedRlEnv,
  metric_prefix: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize lateral body lean without changing the forward-lean target."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, _ = _body_roll_pitch(asset, asset_cfg)
  env.extras["log"][f"Metrics/{metric_prefix}_roll_abs_deg"] = torch.rad2deg(
    torch.mean(torch.abs(roll))
  )
  return torch.square(roll)


def backward_body_pitch_cost(
  env: ManagerBasedRlEnv,
  tolerance: float,
  metric_prefix: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Apply an asymmetric penalty only when a body leans backward."""
  asset: Entity = env.scene[asset_cfg.name]
  _, pitch = _body_roll_pitch(asset, asset_cfg)
  backward_angle = torch.relu(-pitch - tolerance)
  env.extras["log"][f"Metrics/{metric_prefix}_backward_deg"] = (
    torch.rad2deg(torch.mean(backward_angle))
  )
  return torch.square(backward_angle)


def whole_body_com_midfeet_error(
  env: ManagerBasedRlEnv,
  target_forward_offset_at_full_depth: float,
  lateral_scale: float,
  command_name: str,
  command_index: int,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep the whole-robot COM over a depth-dependent point between the feet.

  MuJoCo's subtree COM at the robot root contains the mass-weighted COM of the
  complete articulated robot.  Expressing the COM-minus-midfoot vector in the
  root frame makes the forward target independent of randomized reset yaw.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if len(asset_cfg.site_ids) != 2:
    raise ValueError("COM support term requires exactly two foot sites")
  robot_com_w = asset.data.data.subtree_com[
    :, asset.data.indexing.root_body_id, :
  ]
  midfeet_w = torch.mean(
    asset.data.site_pos_w[:, asset_cfg.site_ids, :], dim=1
  )
  com_from_midfeet_b = quat_apply_inverse(
    asset.data.root_link_quat_w, robot_com_w - midfeet_w
  )
  depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, command_index],
    min=0.0,
    max=1.0,
  )
  target_forward = depth * target_forward_offset_at_full_depth
  forward_error = com_from_midfeet_b[:, 0] - target_forward
  lateral_error = com_from_midfeet_b[:, 1]
  env.extras["log"]["Metrics/crouch_com_forward"] = torch.mean(
    com_from_midfeet_b[:, 0]
  )
  env.extras["log"]["Metrics/crouch_com_target_forward"] = torch.mean(
    target_forward
  )
  env.extras["log"]["Metrics/crouch_com_lateral"] = torch.mean(
    com_from_midfeet_b[:, 1]
  )
  return torch.square(forward_error) + lateral_scale * torch.square(
    lateral_error
  )


def knee_foot_lateral_alignment_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Place each sole laterally below its corresponding knee.

  Hip-roll tracking by itself only opens the knees.  A policy can satisfy it
  while retaining both soles near the centerline, producing the unstable
  "knees out, feet together" pose.  This term measures each knee-to-sole
  vector in the yaw-invariant root frame and directly drives its lateral
  component to zero.  Sagittal knee-over-toe placement remains free so the
  actor can balance during the transition.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if len(asset_cfg.body_ids) != 2 or len(asset_cfg.site_ids) != 2:
    raise ValueError(
      "Knee/foot alignment requires two ordered knee bodies and two foot sites"
    )
  knees_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  feet_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  foot_from_knee_w = feet_w - knees_w
  batch_size, num_legs, _ = foot_from_knee_w.shape
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(
    -1, num_legs, -1
  )
  foot_from_knee_b = quat_apply_inverse(
    root_quat.reshape(-1, 4), foot_from_knee_w.reshape(-1, 3)
  ).reshape(batch_size, num_legs, 3)
  lateral_error = foot_from_knee_b[:, :, 1]

  root_from_foot_w = feet_w - asset.data.root_link_pos_w[:, None, :]
  feet_b = quat_apply_inverse(
    root_quat.reshape(-1, 4), root_from_foot_w.reshape(-1, 3)
  ).reshape(batch_size, num_legs, 3)
  foot_separation = feet_b[:, 0, 1] - feet_b[:, 1, 1]
  knee_separation = (
    foot_separation - lateral_error[:, 0] + lateral_error[:, 1]
  )
  env.extras["log"]["Metrics/crouch_left_foot_from_knee_y"] = torch.mean(
    lateral_error[:, 0]
  )
  env.extras["log"]["Metrics/crouch_right_foot_from_knee_y"] = torch.mean(
    lateral_error[:, 1]
  )
  env.extras["log"]["Metrics/crouch_foot_separation"] = torch.mean(
    foot_separation
  )
  env.extras["log"]["Metrics/crouch_knee_separation"] = torch.mean(
    knee_separation
  )
  return torch.mean(torch.square(lateral_error), dim=1)


def target_foot_separation_error(
  env: ManagerBasedRlEnv,
  standing_separation: float,
  crouched_separation: float,
  command_name: str,
  command_index: int,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track a depth-dependent stance width in the robot root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  if len(asset_cfg.site_ids) != 2:
    raise ValueError("Foot-separation term requires left and right foot sites")
  feet_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
  feet_b = quat_apply_inverse(
    root_quat.reshape(-1, 4),
    (feet_w - asset.data.root_link_pos_w[:, None, :]).reshape(-1, 3),
  ).reshape(asset.data.root_link_pos_w.shape[0], 2, 3)
  separation = feet_b[:, 0, 1] - feet_b[:, 1, 1]
  depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, command_index],
    min=0.0,
    max=1.0,
  )
  target = standing_separation + depth * (
    crouched_separation - standing_separation
  )
  env.extras["log"]["Metrics/crouch_target_foot_separation"] = torch.mean(
    target
  )
  return torch.square(separation - target)


def target_root_height(
  env: ManagerBasedRlEnv,
  start_height: float,
  target_height: float,
  std: float,
  command_name: str,
  command_index: int,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Backward-compatible exponential tracker used by the 23-DoF prototype."""
  error_sq = target_root_height_error(
    env,
    start_height,
    target_height,
    command_name,
    command_index,
    asset_cfg,
  )
  return torch.exp(-error_sq / std**2)


def stationary_base(
  env: ManagerBasedRlEnv,
  linear_std: float,
  angular_std: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Reward a motionless floating base throughout the transition."""
  asset: Entity = env.scene[asset_cfg.name]
  lin_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
  ang_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  return torch.exp(-lin_sq / linear_std**2 - ang_sq / angular_std**2)


def planar_base_velocity_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Penalize the forward/lateral motion inherited from the walking actor."""
  asset: Entity = env.scene[asset_cfg.name]
  planar_velocity = asset.data.root_link_lin_vel_b[:, :2]
  speed = torch.linalg.vector_norm(planar_velocity, dim=1)
  env.extras["log"]["Metrics/crouch_planar_speed"] = torch.mean(speed)
  return torch.sum(torch.square(planar_velocity), dim=1)


def planted_feet_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize horizontal foot motion whenever a sole is in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  foot_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  speed_sq = torch.sum(torch.square(foot_velocity), dim=-1)
  mean_speed = torch.sqrt(torch.clamp(speed_sq, min=0.0))
  env.extras["log"]["Metrics/crouch_foot_speed"] = (
    torch.sum(mean_speed * in_contact) / torch.clamp(torch.sum(in_contact), min=1.0)
  )
  return torch.sum(speed_sq * in_contact, dim=1)


def planted_foot_sagittal_velocity_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize fore-aft sole velocity while allowing stance widening."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  velocity_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  num_envs, num_feet, _ = velocity_w.shape
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1)
  velocity_b = quat_apply_inverse(
    root_quat.reshape(-1, 4), velocity_w.reshape(-1, 3)
  ).reshape(num_envs, num_feet, 3)
  sagittal_velocity = velocity_b[:, :, 0]
  env.extras["log"]["Metrics/crouch_foot_sagittal_speed"] = (
    torch.sum(torch.abs(sagittal_velocity) * in_contact)
    / torch.clamp(torch.sum(in_contact), min=1.0)
  )
  return torch.sum(torch.square(sagittal_velocity) * in_contact, dim=1)


class planted_foot_sagittal_position_cost:
  """Keep each sole at its initial fore-aft takeoff coordinate."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.asset = asset
    self.site_ids = cfg.params["asset_cfg"].site_ids
    self.reference_pos_w = asset.data.site_pos_w[:, self.site_ids, :].clone()
    self.reference_root_quat = asset.data.root_link_quat_w.clone()
    self.capture_reference = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.capture_reference[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del asset_cfg
    current_pos_w = self.asset.data.site_pos_w[:, self.site_ids, :]
    if torch.any(self.capture_reference):
      mask = self.capture_reference
      self.reference_pos_w[mask] = current_pos_w[mask]
      self.reference_root_quat[mask] = self.asset.data.root_link_quat_w[mask]
      self.capture_reference[mask] = False
    displacement_w = current_pos_w - self.reference_pos_w
    num_envs, num_feet, _ = displacement_w.shape
    reference_quat = self.reference_root_quat[:, None, :].expand(
      -1, num_feet, -1
    )
    displacement_b = quat_apply_inverse(
      reference_quat.reshape(-1, 4), displacement_w.reshape(-1, 3)
    ).reshape(num_envs, num_feet, 3)
    sagittal_displacement = displacement_b[:, :, 0]
    env.extras["log"]["Metrics/crouch_foot_sagittal_displacement"] = (
      torch.mean(torch.abs(sagittal_displacement))
    )
    return torch.sum(torch.square(sagittal_displacement), dim=1)


class base_planar_position_drift_cost:
  """Penalize translation of the robot root from its episode start."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.asset = asset
    self.reference_xy = asset.data.root_link_pos_w[:, :2].clone()
    self.capture_reference = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.capture_reference[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del asset_cfg
    current_xy = self.asset.data.root_link_pos_w[:, :2]
    if torch.any(self.capture_reference):
      mask = self.capture_reference
      self.reference_xy[mask] = current_xy[mask]
      self.capture_reference[mask] = False
    displacement = current_xy - self.reference_xy
    env.extras["log"]["Metrics/crouch_base_position_drift"] = torch.mean(
      torch.linalg.vector_norm(displacement, dim=1)
    )
    return torch.sum(torch.square(displacement), dim=1)


class body_yaw_drift_cost:
  """Keep a selected body's world yaw at its episode-start heading."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.asset = asset
    self.body_id = cfg.params["asset_cfg"].body_ids[0]
    self.reference_yaw = torch.zeros(env.num_envs, device=env.device)
    self.capture_reference = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  @staticmethod
  def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(
      2.0 * (w * z + x * y),
      1.0 - 2.0 * (y * y + z * z),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.capture_reference[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del asset_cfg
    yaw = self._yaw(self.asset.data.body_link_quat_w[:, self.body_id, :])
    if torch.any(self.capture_reference):
      mask = self.capture_reference
      self.reference_yaw[mask] = yaw[mask]
      self.capture_reference[mask] = False
    error = torch.atan2(
      torch.sin(yaw - self.reference_yaw),
      torch.cos(yaw - self.reference_yaw),
    )
    env.extras["log"]["Metrics/crouch_torso_yaw_abs_deg"] = torch.rad2deg(
      torch.mean(torch.abs(error))
    )
    return torch.square(error)


def root_yaw_rate_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Penalize yaw angular velocity independently of pitch/roll motion."""
  asset: Entity = env.scene[asset_cfg.name]
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
  env.extras["log"]["Metrics/crouch_yaw_rate_abs_deg_s"] = torch.rad2deg(
    torch.mean(torch.abs(yaw_rate))
  )
  return torch.square(yaw_rate)


class planted_foot_position_cost:
  """Penalize displacement from each sole's position at episode reset.

  A velocity-only cost can be defeated by taking one quick corrective step and
  then standing still. This term keeps both soles at their original planar
  contact locations throughout the complete crouch transition.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.asset = asset
    self.site_ids = cfg.params["asset_cfg"].site_ids
    if len(self.site_ids) != 2:
      raise ValueError("Planted-foot position term requires exactly two sites")
    self.reference_xy = asset.data.site_pos_w[:, self.site_ids, :2].clone()
    self.capture_reference = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.capture_reference[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del asset_cfg
    current_xy = self.asset.data.site_pos_w[:, self.site_ids, :2]
    if torch.any(self.capture_reference):
      mask = self.capture_reference
      self.reference_xy[mask] = current_xy[mask]
      self.capture_reference[mask] = False
    displacement = current_xy - self.reference_xy
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    env.extras["log"]["Metrics/crouch_max_foot_displacement"] = torch.max(
      displacement_norm
    )
    return torch.sum(torch.square(displacement), dim=(1, 2))


def both_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward retaining both support contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.all(sensor.data.found > 0, dim=1).float()


def final_both_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_index: int,
  full_depth_start: float,
  full_depth_value: float = 1.0,
) -> torch.Tensor:
  """Require double support after the stance-widening transition is done."""
  contact = both_feet_contact(env, sensor_name)
  gate = final_depth_gate(
    env,
    command_name=command_name,
    command_index=command_index,
    full_depth_start=full_depth_start,
    full_depth_value=full_depth_value,
  )
  return gate * contact


def final_depth_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_index: int,
  full_depth_start: float,
  full_depth_value: float,
) -> torch.Tensor:
  """Smoothly enable hold-only objectives near the deployed crouch depth."""
  depth = env.command_manager.get_command(command_name)[:, command_index]
  return torch.clamp(
    (depth - full_depth_start)
    / max(full_depth_value - full_depth_start, 1.0e-6),
    min=0.0,
    max=1.0,
  )


def final_stationary_base(
  env: ManagerBasedRlEnv,
  linear_std: float,
  angular_std: float,
  command_name: str,
  command_index: int,
  full_depth_start: float,
  full_depth_value: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Reward remaining motionless after the crouch transition is complete."""
  gate = final_depth_gate(
    env,
    command_name,
    command_index,
    full_depth_start,
    full_depth_value,
  )
  return gate * stationary_base(env, linear_std, angular_std, asset_cfg)


def final_base_motion_cost(
  env: ManagerBasedRlEnv,
  angular_scale: float,
  command_name: str,
  command_index: int,
  full_depth_start: float,
  full_depth_value: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Penalize residual base translation and rotation during the final hold."""
  asset: Entity = env.scene[asset_cfg.name]
  linear_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
  angular_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  gate = final_depth_gate(
    env,
    command_name,
    command_index,
    full_depth_start,
    full_depth_value,
  )
  env.extras["log"]["Metrics/crouch_final_linear_speed"] = torch.mean(
    torch.sqrt(torch.clamp(linear_sq, min=0.0))
  )
  env.extras["log"]["Metrics/crouch_final_angular_speed"] = torch.mean(
    torch.sqrt(torch.clamp(angular_sq, min=0.0))
  )
  return gate * (linear_sq + angular_scale * angular_sq)


def final_joint_velocity_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_index: int,
  full_depth_start: float,
  full_depth_value: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Suppress joint chatter once the target crouch has been reached."""
  gate = final_depth_gate(
    env,
    command_name,
    command_index,
    full_depth_start,
    full_depth_value,
  )
  return gate * torch.sum(torch.square(env.scene[asset_cfg.name].data.joint_vel), dim=1)


def balanced_foot_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float,
) -> torch.Tensor:
  """Reward an even vertical load split between the two feet."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  vertical_force = torch.abs(sensor.data.force[..., 2])
  total = torch.sum(vertical_force, dim=1)
  imbalance = torch.abs(vertical_force[:, 0] - vertical_force[:, 1]) / torch.clamp(
    total, min=1.0
  )
  return torch.exp(-torch.square(imbalance) / std**2)


def joint_velocity_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)
