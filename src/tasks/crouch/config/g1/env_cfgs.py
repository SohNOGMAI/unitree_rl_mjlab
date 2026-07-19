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
  # Match the gap suspension +/-0.50 rad at deployed depth 0.90.
  r"left_hip_roll_joint": 0.50 / 0.90,
  r"right_hip_roll_joint": -0.50 / 0.90,
  # Rotate the flexed shins outward as the stance widens.  At depth 0.90,
  # +/-0.485 rad places each sole almost directly below its own knee instead
  # of leaving both feet together between the abducted knees.
  r"left_hip_yaw_joint": 0.485 / 0.90,
  r"right_hip_yaw_joint": -0.485 / 0.90,
  r"left_knee_joint": 0.95,
  r"right_knee_joint": 0.95,
  # These angles make the sole normal vertical for the widened, toe-out
  # stance above (FK residual tilt <0.1 deg at depth 0.90).
  r"left_ankle_pitch_joint": -0.218 / 0.90,
  r"right_ankle_pitch_joint": -0.218 / 0.90,
  r"left_ankle_roll_joint": -0.0424 / 0.90,
  r"right_ankle_roll_joint": 0.0424 / 0.90,
  r"waist_yaw_joint": 0.0,
  r"waist_roll_joint": 0.0,
  r"waist_pitch_joint": 0.08,
  # Match the gap suspension posture so the crouch-to-hoist handover does not
  # need a second abrupt arm transition.
  r"left_shoulder_pitch_joint": 0.50,
  r"right_shoulder_pitch_joint": 0.50,
  r"left_shoulder_roll_joint": 1.00,
  r"right_shoulder_roll_joint": -1.00,
  r"left_shoulder_yaw_joint": 0.0,
  r"right_shoulder_yaw_joint": 0.0,
  r"left_elbow_joint": 0.50,
  r"right_elbow_joint": 0.50,
  r"left_wrist_roll_joint": 0.0,
  r"right_wrist_roll_joint": 0.0,
  r"left_wrist_pitch_joint": 0.0,
  r"right_wrist_pitch_joint": 0.0,
  r"left_wrist_yaw_joint": 0.0,
  r"right_wrist_yaw_joint": 0.0,
}

# Gap traversal uses a deeper pre-suspension crouch.  This extends beyond the
# previously validated 0.75 command, but remains below the full 1.0 reference.
CROUCH_MAX_DEPTH = 0.90

# Hip yaw changes the support polygon and cannot safely be imposed as a
# feed-forward joint offset while both soles are loaded.  Leave it to the
# learned residual so the policy can unload and reposition one foot before
# reaching the final toe-out target.  All other coordinates retain the smooth
# kinematic reference trajectory.
CROUCH_ACTION_REFERENCE_29DOF = dict(CROUCH_TARGET_29DOF)
CROUCH_ACTION_REFERENCE_29DOF[r"left_hip_yaw_joint"] = 0.0
CROUCH_ACTION_REFERENCE_29DOF[r"right_hip_yaw_joint"] = 0.0


def unitree_g1_crouch_env_cfg(play: bool = False):
  """Track crouch depth while retaining the pretrained standing behavior."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  # Four seconds are used for the transition and the remaining twenty-six for
  # learning to hold the completed crouch without stepping or oscillating.
  cfg.episode_length_s = int(1e9) if play else 30.0
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
    rel_standing_envs=0.05,
    rel_heading_envs=0.0,
    init_velocity_prob=0.0,
    ranges=base_twist.ranges,
    viz=base_twist.viz,
    # Reach the deployed 0.90 depth in four seconds, matching play.py.
    depth_rate=CROUCH_MAX_DEPTH / 4.0,
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
    command_term.func = mdp.crouch_command_with_pose_error
    # A full-scale value would initially mean "walk at 1 m/s" to the walking
    # checkpoint.  The action term below supplies the actual pose reference,
    # so this weak 0.1 signal is enough for the actor to learn residuals without
    # first launching into a walk.
    command_term.params = {
      "command_name": "twist",
      "depth_scale": 0.1,
      "sagittal_position_scale": 2.0,
      "yaw_scale": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    }
    phase_term = cfg.observations[group_name].terms["phase"]
    # The crouch controller has no gait phase.  Reuse the two otherwise-zero
    # slots for planar base velocity so the actor can actively arrest drift
    # while retaining the walking checkpoint's 98-D interface.
    phase_term.func = mdp.base_planar_velocity
    phase_term.params = {"scale": 0.2, "asset_cfg": SceneEntityCfg("robot")}

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
    crouch_target_positions=CROUCH_ACTION_REFERENCE_29DOF,
    command_name="twist",
    command_index=0,
    locked_joint_suffixes=(
      "hip_roll_joint",
      "ankle_roll_joint",
    ),
    always_locked_joint_suffixes=("waist_yaw_joint",),
    lock_residual_at_depth=0.75,
    locked_residual_scale=0.20,
  )

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (-0.03, 0.03),
    "pitch": (-0.03, 0.03),
    "yaw": (0.0, 0.0),
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
    "alive": RewardTermCfg(
      func=mdp.alive,
      weight=20.0,
    ),
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
            ".*_hip_(pitch|roll|yaw)_joint",
            ".*_knee_joint",
            ".*_ankle_(pitch|roll)_joint",
            "waist_pitch_joint",
          ),
        ),
      },
    ),
    "target_hip_roll_error": RewardTermCfg(
      func=mdp.target_joint_posture_error,
      weight=-100.0,
      params={
        "target_positions": CROUCH_TARGET_29DOF,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(".*_hip_roll_joint",)
        ),
      },
    ),
    "target_hip_yaw_error": RewardTermCfg(
      func=mdp.target_joint_posture_error,
      # Hip yaw is only a means of moving each sole below its knee.  Keeping
      # this softer than the Cartesian geometry terms lets the actor choose a
      # contact-safe toe angle instead of twisting a loaded sole in place.
      weight=-30.0,
      params={
        "target_positions": CROUCH_TARGET_29DOF,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(".*_hip_yaw_joint",)
        ),
      },
    ),
    "target_waist_yaw_error": RewardTermCfg(
      func=mdp.target_joint_posture_error,
      weight=-250.0,
      params={
        "target_positions": CROUCH_TARGET_29DOF,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("waist_yaw_joint",)
        ),
      },
    ),
    "target_arm_posture_error": RewardTermCfg(
      func=mdp.target_joint_posture_error,
      weight=-30.0,
      params={
        "target_positions": CROUCH_TARGET_29DOF,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_shoulder_(pitch|roll|yaw)_joint",
            ".*_elbow_joint",
            ".*_wrist_(roll|pitch|yaw)_joint",
          ),
        ),
      },
    ),
    "feet_below_knees": RewardTermCfg(
      func=mdp.knee_foot_lateral_alignment_error,
      weight=-180.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_knee_link", "right_knee_link"),
          site_names=("left_foot", "right_foot"),
        ),
      },
    ),
    "target_foot_separation": RewardTermCfg(
      func=mdp.target_foot_separation_error,
      weight=-180.0,
      params={
        "standing_separation": 0.23,
        "crouched_separation": 0.45,
        "command_name": "twist",
        "command_index": 0,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
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
      weight=6.0,
      params={"linear_std": 0.20, "angular_std": 0.35},
    ),
    "planar_base_velocity": RewardTermCfg(
      func=mdp.planar_base_velocity_cost,
      weight=-30.0,
    ),
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      # Do not forbid the single controlled repositioning step needed to
      # widen the stance.
      weight=8.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "final_both_feet_contact": RewardTermCfg(
      func=mdp.final_both_feet_contact,
      weight=100.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_index": 0,
        "full_depth_start": 0.80,
        "full_depth_value": CROUCH_MAX_DEPTH,
      },
    ),
    "final_stationary_base": RewardTermCfg(
      func=mdp.final_stationary_base,
      weight=30.0,
      params={
        "linear_std": 0.10,
        "angular_std": 0.20,
        "command_name": "twist",
        "command_index": 0,
        "full_depth_start": 0.80,
        "full_depth_value": CROUCH_MAX_DEPTH,
      },
    ),
    "final_base_motion": RewardTermCfg(
      func=mdp.final_base_motion_cost,
      weight=-80.0,
      params={
        "angular_scale": 0.35,
        "command_name": "twist",
        "command_index": 0,
        "full_depth_start": 0.80,
        "full_depth_value": CROUCH_MAX_DEPTH,
      },
    ),
    "final_joint_velocity": RewardTermCfg(
      func=mdp.final_joint_velocity_cost,
      weight=-0.012,
      params={
        "command_name": "twist",
        "command_index": 0,
        "full_depth_start": 0.80,
        "full_depth_value": CROUCH_MAX_DEPTH,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
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
        "target_pitch_at_full_depth": math.radians(16.0),
        "roll_scale": 1.5,
        "command_name": "twist",
        "command_index": 0,
        "metric_prefix": "crouch_torso",
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "torso_roll_level": RewardTermCfg(
      func=mdp.body_roll_level_error,
      weight=-250.0,
      params={
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
        "target_forward_offset_at_full_depth": 0.075,
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
      # Permit a slow outward slide while strongly penalizing a quick step.
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
        ),
      },
    ),
    "planted_foot_sagittal_velocity": RewardTermCfg(
      func=mdp.planted_foot_sagittal_velocity_cost,
      weight=-300.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
        ),
      },
    ),
    "planted_foot_sagittal_position": RewardTermCfg(
      func=mdp.planted_foot_sagittal_position_cost,
      weight=-2000.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("left_foot", "right_foot")
        ),
      },
    ),
    "base_planar_position_drift": RewardTermCfg(
      func=mdp.base_planar_position_drift_cost,
      weight=-1000.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "torso_yaw_drift": RewardTermCfg(
      func=mdp.body_yaw_drift_cost,
      weight=-1000.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "root_yaw_rate": RewardTermCfg(
      func=mdp.root_yaw_rate_cost,
      weight=-100.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "planted_foot_position": RewardTermCfg(
      func=mdp.planted_foot_position_cost,
      # The new target intentionally widens the stance, so the initial foot
      # locations must not dominate the knee/foot alignment objective.
      weight=-0.05,
      params={
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
      weight=-0.20,
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
      # A failure previously cost only about one second of ordinary rewards,
      # so PPO preferred an accurate-looking pose that fell shortly after.
      weight=-10000.0,
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
      # During training, terminate at the beginning of a squat collapse rather
      # than waiting until the torso is already unrecoverably near the floor.
      # Playback keeps the original permissive limit.
      params={"minimum_height": 0.48 if play else 0.60},
    ),
  }

  if not play:
    cfg.terminations["final_support_lost"] = TerminationTermCfg(
      func=mdp.final_support_lost,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_index": 0,
        "depth_threshold": 0.88,
        "grace_time": 0.30,
      },
    )

  if play:
    cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = False

  return cfg
