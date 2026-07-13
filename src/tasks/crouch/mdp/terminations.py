"""Termination terms for the stationary crouch task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
