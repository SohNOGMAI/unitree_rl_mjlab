"""Rate-limited command generator for safe crouch transitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class CrouchDepthCommand(UniformVelocityCommand):
  """Generate a desired depth but expose only a rate-limited current depth."""

  cfg: CrouchDepthCommandCfg

  def __init__(self, cfg: CrouchDepthCommandCfg, env):
    super().__init__(cfg, env)
    self.desired_depth = torch.zeros(self.num_envs, device=self.device)
    self.metrics = {
      "depth_error": torch.zeros(self.num_envs, device=self.device)
    }

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    sample = torch.empty(len(env_ids), device=self.device)
    self.desired_depth[env_ids] = sample.uniform_(*self.cfg.ranges.lin_vel_x)
    stand_mask = sample.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.desired_depth[env_ids[stand_mask]] = 0.0
    # CommandTerm increments command_counter after this method.  A zero count
    # therefore identifies an episode reset rather than an interval resample.
    reset_ids = env_ids[self.command_counter[env_ids] == 0]
    self.vel_command_b[reset_ids] = 0.0

  def _update_command(self) -> None:
    max_delta = self.cfg.depth_rate * self._env.step_dt
    depth_error = self.desired_depth - self.vel_command_b[:, 0]
    self.vel_command_b[:, 0] += torch.clamp(
      depth_error,
      min=-max_delta,
      max=max_delta,
    )
    self.vel_command_b[:, 1:] = 0.0

  def _update_metrics(self) -> None:
    self.metrics["depth_error"] += torch.abs(
      self.desired_depth - self.vel_command_b[:, 0]
    ) / max(self._env.max_episode_length, 1)


@dataclass(kw_only=True)
class CrouchDepthCommandCfg(UniformVelocityCommandCfg):
  """Configuration for a normalized crouch-depth command."""

  depth_rate: float = 1.0 / 3.0

  def build(self, env) -> CrouchDepthCommand:
    return CrouchDepthCommand(self, env)
