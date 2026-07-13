"""Observations used by the command-conditioned crouch policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def zero_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Keep the walking-policy phase slots without inducing a gait cycle."""
  return torch.zeros((env.num_envs, 2), device=env.device)


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
