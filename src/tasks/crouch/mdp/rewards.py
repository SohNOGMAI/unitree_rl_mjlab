"""Reward terms for a planted-feet crouch transition."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT_CFG = SceneEntityCfg("robot")


class target_joint_posture_error:
  """Squared physical joint error to the command-interpolated pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.target = asset.data.default_joint_pos.clone()
    joint_names = asset.joint_names
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


def both_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward retaining both support contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.all(sensor.data.found > 0, dim=1).float()


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
