"""Residual joint-position action used by the crouch fine-tuning task."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.string import resolve_expr


class CrouchResidualJointPositionAction(JointPositionAction):
  """Add a rate-limited crouch reference underneath the learned residual.

  At depth zero this is exactly the regular G1 walking action mapping.  This
  preserves the meaning of the pretrained actor output.  As depth increases,
  the nominal joint target is interpolated toward the crouch pose; PPO only
  has to learn the dynamic balance correction around that reference.
  """

  cfg: CrouchResidualJointPositionActionCfg

  def __init__(self, cfg: CrouchResidualJointPositionActionCfg, env):
    super().__init__(cfg, env)
    target_values = resolve_expr(
      cfg.crouch_target_positions,
      tuple(self.target_names),
      default_val=None,
    )
    if any(value is None for value in target_values):
      missing = [
        name
        for name, value in zip(self.target_names, target_values, strict=True)
        if value is None
      ]
      raise ValueError(f"Crouch target is missing joints: {missing}")
    target = torch.tensor(
      target_values,
      dtype=torch.float32,
      device=self.device,
    )
    assert isinstance(self._offset, torch.Tensor)
    self._crouch_delta = target.unsqueeze(0) - self._offset
    self._locked_action_ids = [
      index
      for index, name in enumerate(self.target_names)
      if any(name.endswith(suffix) for suffix in cfg.locked_joint_suffixes)
    ]
    self._always_locked_action_ids = [
      index
      for index, name in enumerate(self.target_names)
      if any(name.endswith(suffix) for suffix in cfg.always_locked_joint_suffixes)
    ]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    depth = torch.clamp(
      self._env.command_manager.get_command(self.cfg.command_name)[
        :, self.cfg.command_index
      ],
      min=0.0,
      max=1.0,
    ).unsqueeze(1)
    if self._locked_action_ids:
      lock_blend = torch.clamp(
        depth / self.cfg.lock_residual_at_depth, min=0.0, max=1.0
      )
      residual_scale = 1.0 - lock_blend * (1.0 - self.cfg.locked_residual_scale)
      self._raw_actions[:, self._locked_action_ids] *= residual_scale
    if self._always_locked_action_ids:
      # Waist yaw is not a balance actuator for this task.  Allowing the actor
      # to use it produced a repeatable 20--25 degree torso twist during the
      # planted-foot transition, followed by a large yaw impulse at hoist.
      # Keep its learned residual identically zero; the nominal reference and
      # joint PD controller then hold the upper body facing forward.
      self._raw_actions[:, self._always_locked_action_ids] = 0.0
    self._processed_actions = (
      self._raw_actions * self._scale
      + self._offset
      + depth * self._crouch_delta
    )


@dataclass(kw_only=True)
class CrouchResidualJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for command-conditioned crouch residual control."""

  crouch_target_positions: dict[str, float] = field(default_factory=dict)
  command_name: str = "twist"
  command_index: int = 0
  locked_joint_suffixes: tuple[str, ...] = ()
  always_locked_joint_suffixes: tuple[str, ...] = ()
  lock_residual_at_depth: float = 0.75
  # Keep a small ankle/hip correction for lateral balance at full crouch.
  locked_residual_scale: float = 0.20

  def build(self, env) -> CrouchResidualJointPositionAction:
    return CrouchResidualJointPositionAction(self, env)
