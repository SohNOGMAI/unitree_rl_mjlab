"""Batch-evaluate G1 crouch checkpoints at the deployed gap depth."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


def evaluate(
  checkpoints: list[Path], num_envs: int, seconds: float, depth: float
) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg("Unitree-G1-Crouch", play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = 1.0e9
  command_cfg = env_cfg.commands["twist"]
  command_cfg.rel_standing_envs = 0.0
  command_cfg.ranges.lin_vel_x = (depth, depth)
  command_cfg.resampling_time_range = (1.0e6, 1.0e6)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  agent_cfg = load_rl_cfg("Unitree-G1-Crouch")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls("Unitree-G1-Crouch") or MjlabOnPolicyRunner

  robot = env.scene["robot"]
  knee_ids = robot.find_bodies(
    ("left_knee_link", "right_knee_link"), preserve_order=True
  )[0]
  foot_ids = robot.find_sites(
    ("left_foot", "right_foot"), preserve_order=True
  )[0]
  torso_id = robot.find_bodies(("torso_link",), preserve_order=True)[0][0]
  contact_sensor = env.scene["feet_ground_contact"]
  joint_ids = {
    name: robot.joint_names.index(name)
    for name in (
      "left_hip_roll_joint",
      "right_hip_roll_joint",
      "left_hip_yaw_joint",
      "right_hip_yaw_joint",
      "left_shoulder_pitch_joint",
      "right_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "right_shoulder_roll_joint",
      "left_elbow_joint",
      "right_elbow_joint",
    )
  }
  arm_target = {
    "left_shoulder_pitch_joint": 0.50,
    "right_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 1.00,
    "right_shoulder_roll_joint": -1.00,
    "left_elbow_joint": 0.50,
    "right_elbow_joint": 0.50,
  }
  num_steps = round(seconds / env.step_dt)

  print(
    "checkpoint survival depth root_h foot_sep knee_sep "
    "left_dy right_dy torso_pitch_deg torso_roll_deg arm_rms hip_roll_lr hip_yaw_lr "
    "hold_lin_rms hold_ang_rms hold_contact hold_height_std "
    "foot_x_max base_xy_max torso_yaw_max_deg "
    "fail_tilt fail_low mean_fail_s"
  )
  for checkpoint in checkpoints:
    torch.manual_seed(42)
    wrapped.reset()
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint.resolve()),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    obs = wrapped.get_observations()
    initial_feet_w = robot.data.site_pos_w[:, foot_ids, :].clone()
    initial_root_xy = robot.data.root_link_pos_w[:, :2].clone()
    initial_root_quat = robot.data.root_link_quat_w.clone()
    torso_quat = robot.data.body_link_quat_w[:, torso_id, :]
    w, x, y, z = torso_quat.unbind(dim=-1)
    initial_torso_yaw = torch.atan2(
      2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    )
    max_foot_x = torch.zeros(num_envs, device=device)
    max_base_xy = torch.zeros(num_envs, device=device)
    max_torso_yaw = torch.zeros(num_envs, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    hold_count = torch.zeros(num_envs, device=device)
    hold_linear_sq = torch.zeros(num_envs, device=device)
    hold_angular_sq = torch.zeros(num_envs, device=device)
    hold_contact = torch.zeros(num_envs, device=device)
    hold_height = torch.zeros(num_envs, device=device)
    hold_height_sq = torch.zeros(num_envs, device=device)
    failure_time = torch.zeros(num_envs, device=device)
    failure_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    failure_counts = {
      name: torch.zeros((), device=device)
      for name in env.termination_manager.active_terms
    }
    with torch.no_grad():
      for step_index in range(num_steps):
        obs, _, dones, _ = wrapped.step(policy(obs))
        just_failed = alive & (dones != 0)
        failure_time[just_failed] = (step_index + 1) * env.step_dt
        failure_seen |= just_failed
        for name in failure_counts:
          failure_counts[name] += torch.count_nonzero(
            just_failed & env.termination_manager.get_term(name)
          )
        alive &= dones == 0
        foot_displacement_w = robot.data.site_pos_w[:, foot_ids, :] - initial_feet_w
        reference_quat = initial_root_quat[:, None, :].expand(-1, 2, -1)
        foot_displacement_b = quat_apply_inverse(
          reference_quat.reshape(-1, 4), foot_displacement_w.reshape(-1, 3)
        ).reshape(num_envs, 2, 3)
        max_foot_x = torch.maximum(
          max_foot_x, torch.amax(torch.abs(foot_displacement_b[:, :, 0]), dim=1)
        )
        max_base_xy = torch.maximum(
          max_base_xy,
          torch.linalg.vector_norm(
            robot.data.root_link_pos_w[:, :2] - initial_root_xy, dim=1
          ),
        )
        torso_quat = robot.data.body_link_quat_w[:, torso_id, :]
        w, x, y, z = torso_quat.unbind(dim=-1)
        torso_yaw = torch.atan2(
          2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )
        yaw_error = torch.atan2(
          torch.sin(torso_yaw - initial_torso_yaw),
          torch.cos(torso_yaw - initial_torso_yaw),
        )
        max_torso_yaw = torch.maximum(max_torso_yaw, torch.abs(yaw_error))
        current_depth = env.command_manager.get_command("twist")[:, 0]
        hold_mask = alive & (current_depth >= depth - 0.01)
        hold_weight = hold_mask.float()
        linear_sq = torch.sum(torch.square(robot.data.root_link_lin_vel_b), dim=1)
        angular_sq = torch.sum(
          torch.square(robot.data.root_link_ang_vel_b), dim=1
        )
        height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        assert contact_sensor.data.found is not None
        double_contact = torch.all(contact_sensor.data.found > 0, dim=1).float()
        hold_count += hold_weight
        hold_linear_sq += hold_weight * linear_sq
        hold_angular_sq += hold_weight * angular_sq
        hold_contact += hold_weight * double_contact
        hold_height += hold_weight * height
        hold_height_sq += hold_weight * torch.square(height)

    sample = alive if torch.any(alive) else torch.ones_like(alive)
    knees_w = robot.data.body_link_pos_w[:, knee_ids, :]
    feet_w = robot.data.site_pos_w[:, foot_ids, :]
    root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    foot_from_knee_b = quat_apply_inverse(
      root_quat.reshape(-1, 4), (feet_w - knees_w).reshape(-1, 3)
    ).reshape(num_envs, 2, 3)
    root_from_foot_b = quat_apply_inverse(
      root_quat.reshape(-1, 4),
      (feet_w - robot.data.root_link_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(num_envs, 2, 3)
    root_from_knee_b = quat_apply_inverse(
      root_quat.reshape(-1, 4),
      (knees_w - robot.data.root_link_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(num_envs, 2, 3)
    foot_sep = root_from_foot_b[:, 0, 1] - root_from_foot_b[:, 1, 1]
    knee_sep = root_from_knee_b[:, 0, 1] - root_from_knee_b[:, 1, 1]
    depth = env.command_manager.get_command("twist")[:, 0]
    root_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    joint_pos = robot.data.joint_pos
    projected_gravity_b = quat_apply_inverse(
      robot.data.body_link_quat_w[:, torso_id, :],
      robot.data.gravity_vec_w,
    )
    torso_roll = torch.atan2(
      -projected_gravity_b[:, 1], -projected_gravity_b[:, 2]
    )
    torso_pitch = torch.atan2(
      projected_gravity_b[:, 0], -projected_gravity_b[:, 2]
    )
    arm_error_sq = torch.zeros(num_envs, device=device)
    for name, target in arm_target.items():
      arm_error_sq += torch.square(joint_pos[:, joint_ids[name]] - target)
    arm_rms = torch.sqrt(arm_error_sq / len(arm_target))

    valid_hold = hold_count > 0
    hold_denom = torch.clamp(hold_count, min=1.0)
    hold_lin_rms = torch.sqrt(hold_linear_sq / hold_denom)
    hold_ang_rms = torch.sqrt(hold_angular_sq / hold_denom)
    hold_contact_ratio = hold_contact / hold_denom
    hold_height_mean = hold_height / hold_denom
    hold_height_std = torch.sqrt(
      torch.clamp(hold_height_sq / hold_denom - torch.square(hold_height_mean), min=0.0)
    )

    mean = lambda value: torch.mean(value[sample]).item()
    survival = torch.mean(alive.float()).item()
    print(
      f"{checkpoint.name} {survival:.3f} {mean(depth):.3f} "
      f"{mean(root_height):.3f} {mean(foot_sep):.3f} {mean(knee_sep):.3f} "
      f"{mean(foot_from_knee_b[:, 0, 1]):+.3f} "
      f"{mean(foot_from_knee_b[:, 1, 1]):+.3f} "
      f"{torch.rad2deg(torch.mean(torso_pitch[sample])).item():.2f} "
      f"{torch.rad2deg(torch.mean(torch.abs(torso_roll[sample]))).item():.2f} "
      f"{mean(arm_rms):.3f} "
      f"({mean(joint_pos[:, joint_ids['left_hip_roll_joint']]):+.3f},"
      f"{mean(joint_pos[:, joint_ids['right_hip_roll_joint']]):+.3f}) "
      f"({mean(joint_pos[:, joint_ids['left_hip_yaw_joint']]):+.3f},"
      f"{mean(joint_pos[:, joint_ids['right_hip_yaw_joint']]):+.3f}) "
      f"{torch.mean(hold_lin_rms[valid_hold]).item():.3f} "
      f"{torch.mean(hold_ang_rms[valid_hold]).item():.3f} "
      f"{torch.mean(hold_contact_ratio[valid_hold]).item():.3f} "
      f"{torch.mean(hold_height_std[valid_hold]).item():.3f} "
      f"{mean(max_foot_x):.3f} "
      f"{mean(max_base_xy):.3f} "
      f"{mean(max_torso_yaw) * 180.0 / torch.pi:.2f} "
      f"{failure_counts.get('fell_over', torch.zeros((), device=device)).item() / num_envs:.3f} "
      f"{failure_counts.get('root_too_low', torch.zeros((), device=device)).item() / num_envs:.3f} "
      f"{torch.mean(failure_time[failure_seen]).item() if torch.any(failure_seen) else seconds:.2f}"
    )

  wrapped.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoints", nargs="+", type=Path)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--seconds", type=float, default=8.0)
  parser.add_argument("--depth", type=float, default=0.90)
  args = parser.parse_args()
  evaluate(args.checkpoints, args.num_envs, args.seconds, args.depth)


if __name__ == "__main__":
  main()
