"""Termination terms for the stationary crouch task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def root_height_below(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height < minimum_height


class final_support_lost:
  """Terminate only after sustained loss of a foot at the final depth."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.counter[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_index: int,
    depth_threshold: float,
    grace_time: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    depth = env.command_manager.get_command(command_name)[:, command_index]
    both_contact = torch.all(sensor.data.found > 0, dim=1)
    unsupported = (depth >= depth_threshold) & ~both_contact
    self.counter = torch.where(
      unsupported, self.counter + 1, torch.zeros_like(self.counter)
    )
    grace_steps = max(round(grace_time / env.step_dt), 1)
    return self.counter >= grace_steps
