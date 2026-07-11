"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

class WireAssistWrapper:
    """
    【固定アンカー・非対称張力制御版】
    ロボットの歩行を阻害せず、倒れる方向の物理的制約（前方にアンカーがある時は前に引くと危ない等）を考慮した高度なワイヤーアシスト。
    """
    def __init__(self, env, anchor_pos=[3.0, 2.5], 
                 base_tension=10.0, kp_pitch=40.0, kz_com=150.0,
                 pitch_threshold=0.08, max_tension=80.0,
                 lin_vel_x=0.5, lin_vel_y=0.0, yaw_vel=0.0):
        self.env = env
        
        # ワイヤーの端点（ワールド座標の完全固定点: X, Y, Z）
        self.anchor_pos = np.array([anchor_pos[0], 0.0, anchor_pos[1]])
        
        # 張力パラメータ（全体的に非常にマイルドに設定）
        self.base_tension = base_tension
        self.kp_pitch = kp_pitch # 姿勢回復ゲイン
        self.kz_com = kz_com     # 沈み込み回復ゲイン
        self.pitch_threshold = pitch_threshold
        self.max_tension = max_tension # 絶対にロボットを吹っ飛ばさない上限張力 (約8kg分)
        
        self.lin_vel_x = lin_vel_x
        self.lin_vel_y = lin_vel_y
        self.yaw_vel = yaw_vel
        
        self.z_ref = None
        self.prev_pitch = 0.0
        self.step_counter = 0
        
        self.mj_model = getattr(env.unwrapped, "model", None) or getattr(getattr(env.unwrapped, "sim", None), "model", None)
        self.mj_data = getattr(env.unwrapped, "data", None) or getattr(getattr(env.unwrapped, "sim", None), "data", None)
        native_model = getattr(self.mj_model, "mj_model", None) or getattr(getattr(env.unwrapped, "sim", None), "mj_model", None)
        
        self.dt = getattr(env.unwrapped, "step_dt", 0.02)
        self.body_id = -1

        if native_model is None or self.mj_data is None:
            print("⚠️ [WireAssist] MuJoCoのnative modelが見つかりません。")
        else:
            for i in range(native_model.nbody):
                name = mj.mj_id2name(native_model, mj.mjtObj.mjOBJ_BODY, i)
                if name:
                    name_lower = name.lower()
                    if "torso" in name_lower or "pelvis" in name_lower or "trunk" in name_lower:
                        self.body_id = i
                        print(f"✅ [WireAssist] 胴体を発見！ パーツ名: '{name}'")
                        break

    def _force_velocity_command(self):
        try:
            cm = getattr(self.env.unwrapped, "command_manager", None)
            if cm is None: return False
            command = cm.get_command("twist")
            if command is None: return False
            target = torch.tensor([self.lin_vel_x, self.lin_vel_y, self.yaw_vel], device=command.device, dtype=command.dtype)
            with torch.no_grad():
                command[:] = target
            return True
        except Exception:
            return False

    def _get_robot_state(self):
        if self.body_id == -1: return None
        import torch
        
        quat_data = self.mj_data.xquat
        quat_array = quat_data.detach().cpu().numpy() if hasattr(quat_data, "cpu") else quat_data
        quat = quat_array[0, self.body_id] if len(quat_array.shape) == 3 else quat_array[self.body_id]

        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        pitch = r.as_euler("xyz", degrees=False)[1]
        
        pos_data = self.mj_data.xpos
        pos_array = pos_data.detach().cpu().numpy() if hasattr(pos_data, "cpu") else pos_data
        torso_pos = pos_array[0, self.body_id] if len(pos_array.shape) == 3 else pos_array[self.body_id]
        
        try:
            com_data = self.mj_data.subtree_com
            com_array = com_data.detach().cpu().numpy() if hasattr(com_data, "cpu") else com_data
            com_pos = com_array[0, 1] if len(com_array.shape) == 3 else com_array[1]
        except:
            com_pos = torso_pos

        return torso_pos, pitch, com_pos

    def step(self, action):
        self._force_velocity_command()
        self.step_counter += 1
        
        state = self._get_robot_state()
        if state is not None:
            torso_pos, pitch, com_pos = state
            
            if self.z_ref is None:
                self.z_ref = com_pos[2]
                
            # ワイヤーベクトルの計算
            direction = self.anchor_pos - torso_pos
            current_length = np.linalg.norm(direction)
            u_vec = direction / current_length if current_length > 1e-3 else np.array([0.0, 0.0, 1.0])
            
            # --- 高度な非対称・張力制御アルゴリズム ---
            f_target = self.base_tension
            
            # アンカーがロボットの前方にあるか後方にあるか
            is_anchor_forward = direction[0] > 0
            
            if is_anchor_forward:
                # 【アンカーが前方にある時】
                if pitch < -self.pitch_threshold:
                    # 後ろに倒れそうな時だけ、前に引き起こす
                    f_target += self.kp_pitch * (-pitch - self.pitch_threshold)
                elif pitch > self.pitch_threshold:
                    # 前に倒れそうな時に引っ張ると余計転ぶため、あえて張力を抜く（スラック制御）
                    f_target = max(0.0, f_target - 10.0 * (pitch - self.pitch_threshold))
            else:
                # 【アンカーが後方にある時】(ロボットが通り過ぎた後)
                if pitch > self.pitch_threshold:
                    # 前に倒れそうな時に、後ろに引き起こす
                    f_target += self.kp_pitch * (pitch - self.pitch_threshold)
                elif pitch < -self.pitch_threshold:
                    # 後ろに倒れそうな時は張力を抜く
                    f_target = max(0.0, f_target - 10.0 * (-pitch - self.pitch_threshold))
                    
            # 重心落下補償（しゃがみ込み、段差登攀時のサポート）
            delta_z = self.z_ref - com_pos[2]
            if delta_z > 0.05:  
                # 高度2.5mのアンカーを活かし、Z方向への引き上げ力を強化
                f_target += self.kz_com * delta_z
                
            # 歩行を邪魔しないための絶対安全リミッター
            tension = np.clip(f_target, 0.0, self.max_tension)
                
            # 外力の印加
            force_vec = tension * u_vec
            force_6d = np.zeros(6, dtype=np.float32)
            force_6d[0:3] = force_vec
            
            import torch
            xfrc = self.mj_data.xfrc_applied
            if hasattr(xfrc, "cpu"):
                xfrc_target = xfrc[0, self.body_id] if len(xfrc.shape) == 3 else xfrc[self.body_id]
                xfrc_target.copy_(torch.tensor(force_6d, device=xfrc.device))
            else:
                if len(xfrc.shape) == 3: xfrc[0, self.body_id] = force_6d
                else: xfrc[self.body_id] = force_6d

            if self.step_counter % 50 == 0:
                print(f"🔗 [WireAssist] 張力: {tension:.1f}N | 力ベクトル: [{force_vec[0]:.1f}, {force_vec[1]:.1f}, {force_vec[2]:.1f}] | Pitch: {np.degrees(pitch):.1f}度")

        return self.env.step(action)

    def reset(self, **kwargs):
        self._force_velocity_command()
        self.z_ref = None
        self.prev_pitch = 0.0
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)

@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False

  _demo_mode: tyro.conf.Suppress[bool] = False

def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError("Tracking tasks require motion file")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
    else:
      resume_path, _ = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
    log_dir = resume_path.parent

  if cfg.num_envs is not None: env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None: env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None: env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # =========================================================================
  # 固定アンカー設定 (X=3.0mの前方、高さZ=2.5mからアシスト)
  # 張力の上限を80Nに抑え、歩行を阻害しない設計に変更
  # =========================================================================
  env = WireAssistWrapper(env, anchor_pos=[3.0, 2.5], base_tension=10.0, max_tension=80.0, lin_vel_x=0.5)
  
  if TRAINED_MODE and cfg.video:
    env = VideoRecorder(env, video_folder=log_dir / "videos" / "play", step_trigger=lambda step: step == 0, video_length=cfg.video_length, disable_logger=True)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":
      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor: return torch.zeros(action_shape, device=env.unwrapped.device)
      policy = PolicyZero()
    else:
      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor: return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native": NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser": ViserPlayViewer(env, policy).run()
  else: raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  env.close()

def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(tyro.extras.literal_type_from_choices(all_tasks), add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS)
  args = tyro.cli(PlayConfig, args=remaining_args, default=PlayConfig(), prog=sys.argv[0] + f" {chosen_task}", config=mjlab.TYRO_FLAGS)
  run_play(chosen_task, args)

if __name__ == "__main__":
  main()