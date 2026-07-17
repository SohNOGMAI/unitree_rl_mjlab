"""Unitree G1 29-DoF command-conditioned crouch task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.crouch import mdp
from src.tasks.velocity import mdp as velocity_mdp
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


CROUCH_TARGET_29DOF = {
  # This sagittal chain sums to -0.15 rad at full depth.  With level feet the
  # pelvis therefore leans forward instead of retreating behind the heels.
  # At the deployed depth of 0.75, forward kinematics gives approximately
  # +6.4 deg pelvis pitch, +9.9 deg torso pitch and a +4.5 cm COM offset.
  r"left_hip_pitch_joint": -0.55,
  r"right_hip_pitch_joint": -0.55,
  r"left_hip_roll_joint": 0.0,
  r"right_hip_roll_joint": 0.0,
  r"left_hip_yaw_joint": 0.0,
  r"right_hip_yaw_joint": 0.0,
  r"left_knee_joint": 0.95,
  r"right_knee_joint": 0.95,
  r"left_ankle_pitch_joint": -0.55,
  r"right_ankle_pitch_joint": -0.55,
  r"left_ankle_roll_joint": 0.0,
  r"right_ankle_roll_joint": 0.0,
  r"waist_yaw_joint": 0.0,
  r"waist_roll_joint": 0.0,
  r"waist_pitch_joint": 0.08,
  r"left_shoulder_pitch_joint": 0.80,
  r"right_shoulder_pitch_joint": 0.80,
  r"left_shoulder_roll_joint": 0.30,
  r"right_shoulder_roll_joint": -0.30,
  r"left_shoulder_yaw_joint": 0.0,
  r"right_shoulder_yaw_joint": 0.0,
  r"left_elbow_joint": 0.60,
  r"right_elbow_joint": 0.60,
  r"left_wrist_roll_joint": 0.0,
  r"right_wrist_roll_joint": 0.0,
  r"left_wrist_pitch_joint": 0.0,
  r"right_wrist_pitch_joint": 0.0,
  r"left_wrist_yaw_joint": 0.0,
  r"right_wrist_yaw_joint": 0.0,
}

# The deepest command that passed the deterministic 10-second stability test.
# A value of 0.75 lowered the pelvis by about 0.17 m without a reset; 1.0 has
# not yet passed that safety gate and must not be sent by the gap controller.
CROUCH_MAX_DEPTH = 0.75


def unitree_g1_crouch_env_cfg(play: bool = False):
  """Track crouch depth while retaining the pretrained standing behavior."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.episode_length_s = int(1e9) if play else 12.0
  cfg.curriculum = {}

  # Random exploratory actions create more contacts than normal walking.
  cfg.sim.njmax = 512
  cfg.sim.nconmax = 96

  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  twist = mdp.CrouchDepthCommandCfg(
    # One stand/full-crouch choice per episode.  Intermediate depths are still
    # visited during the eight-second rate-limited transition.
    resampling_time_range=(12.0, 12.0),
    debug_vis=False,
    entity_name=base_twist.entity_name,
    heading_command=False,
    heading_control_stiffness=base_twist.heading_control_stiffness,
    rel_standing_envs=0.20,
    rel_heading_envs=0.0,
    init_velocity_prob=0.0,
    ranges=base_twist.ranges,
    viz=base_twist.viz,
    # Eight seconds from standing to full depth.  The unassisted reference
    # becomes unstable when forced through this transition in four seconds.
    depth_rate=1.0 / 8.0,
  )
  cfg.commands["twist"] = twist
  # Reinterpret command[0] as crouch depth: 0=stand, 1=full crouch.
  # The walking warm start already stands well.  Spend most samples learning
  # nonzero crouch depths while retaining enough zero-depth samples to prevent
  # catastrophic forgetting of the edge stance.
  twist.ranges.lin_vel_x = (CROUCH_MAX_DEPTH, CROUCH_MAX_DEPTH)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.ranges.heading = None

  # Preserve the walking actor's 98-D interface but disable its gait clock.
  for group_name in ("actor", "critic"):
    command_term = cfg.observations[group_name].terms["command"]
    command_term.func = mdp.scaled_crouch_command
    # A full-scale value would initially mean "walk at 1 m/s" to the walking
    # checkpoint.  The action term below supplies the actual pose reference,
    # so this weak 0.1 signal is enough for the actor to learn residuals without
    # first launching into a walk.
    command_term.params = {"command_name": "twist", "scale": 0.1}
    phase_term = cfg.observations[group_name].terms["phase"]
    phase_term.func = mdp.zero_phase
    phase_term.params = {}

  # Keep the exact 29-D walking action interface while adding a nominal crouch
  # trajectory underneath it.  At command depth zero this is bit-for-bit the
  # original walking action transformation.
  base_action = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = mdp.CrouchResidualJointPositionActionCfg(
    entity_name=base_action.entity_name,
    actuator_names=base_action.actuator_names,
    scale=base_action.scale,
    offset=base_action.offset,
    preserve_order=base_action.preserve_order,
    clip=base_action.clip,
    use_default_offset=base_action.use_default_offset,
    crouch_target_positions=CROUCH_TARGET_29DOF,
    command_name="twist",
    command_index=0,
  )

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (0.0, 0.0),
    "roll": (-0.03, 0.03),
    "pitch": (-0.03, 0.03),
    "yaw": (-0.10, 0.10),
  }
  reset_base.params["velocity_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.02, 0.02),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.05, 0.05),
  }
  reset_joints = cfg.events["reset_robot_joints"]
  reset_joints.params["position_range"] = (-0.03, 0.03)
  reset_joints.params["velocity_range"] = (-0.05, 0.05)

  if "push_robot" in cfg.events:
    # Learn the planted-feet transition first.  Disturbance robustness can be
    # added in a later fine-tuning stage after a valid crouch exists.
    cfg.events.pop("push_robot")

  cfg.rewards = {
    "target_joint_posture_error": RewardTermCfg(
      func=mdp.target_joint_posture_error,
      weight=-12.0,
      params={
        "target_positions": CROUCH_TARGET_29DOF,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            "waist_pitch_joint",
          ),
        ),
      },
    ),
    "target_root_height_error": RewardTermCfg(
      func=mdp.target_root_height_error,
      weight=-1000.0,
      params={
        "start_height": 0.78,
        # The joint reference supplies most of the crouch depth; this modest
        # height target discourages collapse without forcing a rapid descent.
        "target_height": 0.73,
        "command_name": "twist",
        "command_index": 0,
      },
    ),
    "stationary_base": RewardTermCfg(
      func=mdp.stationary_base,
      weight=3.0,
      params={"linear_std": 0.20, "angular_std": 0.35},
    ),
    "planar_base_velocity": RewardTermCfg(
      func=mdp.planar_base_velocity_cost,
      weight=-12.0,
    ),
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=1.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "balanced_foot_forces": RewardTermCfg(
      func=mdp.balanced_foot_forces,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "std": 0.25},
    ),
    "target_torso_attitude": RewardTermCfg(
      func=mdp.target_body_attitude_error,
      # Stage 2: once the policy can crouch without falling, make the intended
      # forward lean materially more valuable than the inherited upright
      # walking solution.
      weight=-100.0,
      params={
        # 13.2 deg at command depth 1.0 becomes about 9.9 deg at the deployed
        # maximum depth of 0.75.
        "target_pitch_at_full_depth": math.radians(13.2),
        "roll_scale": 1.5,
        "command_name": "twist",
        "command_index": 0,
        "metric_prefix": "crouch_torso",
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "target_pelvis_attitude": RewardTermCfg(
      func=mdp.target_body_attitude_error,
      weight=-60.0,
      params={
        # 8.6 deg at depth 1.0 becomes about 6.4 deg at depth 0.75.
        "target_pitch_at_full_depth": math.radians(8.6),
        "roll_scale": 2.0,
        "command_name": "twist",
        "command_index": 0,
        "metric_prefix": "crouch_pelvis",
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),
    "backward_torso": RewardTermCfg(
      func=mdp.backward_body_pitch_cost,
      weight=-60.0,
      params={
        "tolerance": math.radians(1.0),
        "metric_prefix": "crouch_torso",
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "whole_body_com_support": RewardTermCfg(
      func=mdp.whole_body_com_midfeet_error,
      weight=-300.0,
      params={
        # The kinematic reference yields +4.5 cm at depth 0.75.
        "target_forward_offset_at_full_depth": 0.060,
        "lateral_scale": 2.0,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
        ),
      },
    ),
    "planted_feet": RewardTermCfg(
      func=mdp.planted_feet_cost,
      weight=-6.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
        ),
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=velocity_mdp.body_angular_velocity_penalty,
      weight=-0.15,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))
      },
    ),
    "joint_velocity": RewardTermCfg(
      func=mdp.joint_velocity_cost,
      weight=-0.003,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate": RewardTermCfg(
      func=velocity_mdp.action_rate_l2,
      weight=-0.12,
    ),
    "joint_acceleration": RewardTermCfg(
      func=velocity_mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
    "joint_limits": RewardTermCfg(
      func=velocity_mdp.joint_pos_limits,
      weight=-10.0,
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "is_terminated": RewardTermCfg(
      func=velocity_mdp.is_terminated,
      weight=-1000.0,
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(
      func=velocity_mdp.time_out,
      time_out=True,
    ),
    "fell_over": TerminationTermCfg(
      func=velocity_mdp.bad_orientation,
      params={"limit_angle": math.radians(50.0)},
    ),
    "root_too_low": TerminationTermCfg(
      func=mdp.root_height_below,
      params={"minimum_height": 0.48},
    ),
  }

  if play:
    cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = False

  return cfg
