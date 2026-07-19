"""Observations used by the command-conditioned crouch policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def zero_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Keep the walking-policy phase slots without inducing a gait cycle."""
  return torch.zeros((env.num_envs, 2), device=env.device)


def base_planar_velocity(
  env: ManagerBasedRlEnv,
  scale: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Expose planar drift in the two legacy gait-phase observation slots."""
  asset: Entity = env.scene[asset_cfg.name]
  return scale * asset.data.root_link_lin_vel_b[:, :2]


def scaled_crouch_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float,
) -> torch.Tensor:
  """Encode crouch depth weakly in the old forward-command observation slot.

  The command manager retains an unscaled depth in [0, 1] for rewards.  The
  pretrained walking actor sees only [0, 0.1], which keeps its initial behavior
  close to standing instead of interpreting full crouch as a 1 m/s walk.
  """
  depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, 0],
    min=0.0,
    max=1.0,
  )
  command = torch.zeros((env.num_envs, 3), device=env.device)
  command[:, 0] = depth * scale
  return command


def crouch_command_with_pose_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  depth_scale: float,
  sagittal_position_scale: float,
  yaw_scale: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Expose depth, fore-aft position error and heading error in three slots."""
  asset: Entity = env.scene[asset_cfg.name]
  depth = torch.clamp(
    env.command_manager.get_command(command_name)[:, 0], min=0.0, max=1.0
  )
  root_pos = asset.data.root_link_pos_w - env.scene.env_origins
  quat = asset.data.root_link_quat_w
  w, x, y, z = quat.unbind(dim=-1)
  yaw = torch.atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z),
  )
  observation = torch.zeros((env.num_envs, 3), device=env.device)
  observation[:, 0] = depth * depth_scale
  observation[:, 1] = root_pos[:, 0] * sagittal_position_scale
  observation[:, 2] = yaw * yaw_scale
  return observation
