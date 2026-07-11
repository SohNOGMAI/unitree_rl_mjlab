"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

# === 【追加】ワイヤーアシスト用のライブラリ ===
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
# ============================================

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

class WireAssistWrapper:
    """ロボットの姿勢を監視しつつ、毎ステップで前進コマンドを強制的に上書きするラッパー。"""
    def __init__(self, env, pitch_threshold=0.30, tension=300.0, lin_vel_x=0.5, lin_vel_y=0.0, yaw_vel=0.0):
        self.env = env
        self.pitch_threshold = pitch_threshold
        self.tension = tension
        self.lin_vel_x = lin_vel_x
        self.lin_vel_y = lin_vel_y
        self.yaw_vel = yaw_vel
        self.step_count = 0  # デバッグログ用カウンター

        vec_fwd = np.array([0.5, 0.0, 1.0])
        self.dir_fwd = vec_fwd / np.linalg.norm(vec_fwd)

        vec_bwd = np.array([-0.5, 0.0, 1.0])
        self.dir_bwd = vec_bwd / np.linalg.norm(vec_bwd)

        self.mj_model = getattr(env.unwrapped, "model", None) or getattr(getattr(env.unwrapped, "sim", None), "model", None)
        self.mj_data = getattr(env.unwrapped, "data", None) or getattr(getattr(env.unwrapped, "sim", None), "data", None)
        native_model = getattr(self.mj_model, "mj_model", None) or getattr(getattr(env.unwrapped, "sim", None), "mj_model", None)

        self.body_id = -1
        if native_model is None or self.mj_data is None:
            print("⚠️ [WireAssist] MuJoCoのnative modelが見つかりません。")
        else:
            # 最初に、既知の名前で検索（完全マッチと前方マッチ）
            candidate_names = ["pelvis", "torso", "torso_link", "trunk", "base_link", "base", "waist", "robot/pelvis"]
            for name in candidate_names:
                bid = mj.mj_name2id(native_model, mj.mjtObj.mjOBJ_BODY, name)
                if bid != -1:
                    self.body_id = bid
                    print(f"✅ [WireAssist] 接続成功！ 監視対象パーツ名: '{name}' (ID: {self.body_id})")
                    break

            # 見つからなかった場合は、存在するボディをすべてリストアップして最適なものを選ぶ
            if self.body_id == -1:
                print("⚠️ [WireAssist] 既知の名前で見つかりませんでした。モデル内のボディを検索中...")
                # 優先順位を設定: pelvisが最優先
                priority_keywords = ["pelvis", "torso", "body", "waist", "abdomen", "spine"]
                best_candidate = None
                best_priority = -1
                
                for i in range(native_model.nbody):
                    body_name = mj.mj_id2name(native_model, mj.mjtObj.mjOBJ_BODY, i)
                    if body_name and len(body_name) > 0:
                        print(f"  - Body {i}: {body_name}")
                        # キーワードマッチングで優先度を決定
                        for priority, kw in enumerate(priority_keywords):
                            if kw in body_name.lower():
                                if priority > best_priority:
                                    best_priority = priority
                                    best_candidate = (i, body_name)
                                break

                if best_candidate is not None:
                    self.body_id = best_candidate[0]
                    print(f"✅ [WireAssist] 接続成功！ 監視対象パーツ: '{best_candidate[1]}' (ID: {self.body_id})")
                else:
                    print("⚠️ [WireAssist] 適切な胴体が見つかりません。ワイヤーは作動しません。")

    def _force_velocity_command(self):
      try:
        cm = getattr(self.env.unwrapped, "command_manager", None)
        if cm is None:
          return False
        command = cm.get_command("twist")
        if command is None:
          return False
        target = torch.tensor(
          [self.lin_vel_x, self.lin_vel_y, self.yaw_vel],
          device=command.device,
          dtype=command.dtype,
        )
        with torch.no_grad():
          command[:] = target
        return True
      except Exception as exc:
        print(f"⚠️ [WireAssist] 速度コマンドの上書きに失敗: {exc}")
        return False

    def step(self, action):
        self._force_velocity_command()
        self.step_count += 1

        if self.body_id != -1:
            quat_data = self.mj_data.xquat
            quat_array = quat_data.detach().cpu().numpy() if hasattr(quat_data, "cpu") else quat_data
            quat = quat_array[0, self.body_id] if len(quat_array.shape) == 3 else quat_array[self.body_id]

            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            pitch = r.as_euler("xyz", degrees=False)[1]

            force_3d = np.zeros(3, dtype=np.float32)

            if pitch > self.pitch_threshold:
                force_3d = self.tension * self.dir_fwd
                force_norm = np.linalg.norm(force_3d)
                print(f"🔴 [WireAssist] 前方転倒の危機 (Pitch: {pitch:.4f}rad / {np.degrees(pitch):.2f}°) -> ワイヤー作動! Force: {force_norm:.2f}N, Direction: {force_3d}")
            elif pitch < -self.pitch_threshold:
                force_3d = self.tension * self.dir_bwd
                force_norm = np.linalg.norm(force_3d)
                print(f"🔴 [WireAssist] 後方転倒の危機 (Pitch: {pitch:.4f}rad / {np.degrees(pitch):.2f}°) -> ワイヤー作動! Force: {force_norm:.2f}N, Direction: {force_3d}")
            
            # 50フレームごとにピッチ角度をデバッグ出力
            if self.step_count % 50 == 0:
                print(f"[WireAssist Debug] Step {self.step_count}: Pitch = {pitch:.4f}rad ({np.degrees(pitch):.2f}°), Threshold = {self.pitch_threshold:.4f}rad ({np.degrees(self.pitch_threshold):.2f}°)")

            force_6d = np.zeros(6, dtype=np.float32)
            force_6d[0:3] = force_3d

            xfrc = self.mj_data.xfrc_applied
            try:
                if hasattr(xfrc, "cpu"):
                    xfrc_target = xfrc[0, self.body_id] if len(xfrc.shape) == 3 else xfrc[self.body_id]
                    xfrc_target.copy_(torch.tensor(force_6d, device=xfrc.device))
                else:
                    if len(xfrc.shape) == 3:
                        xfrc[0, self.body_id] = force_6d
                    else:
                        xfrc[self.body_id] = force_6d
                # 力が適用されたか確認
                if np.linalg.norm(force_3d) > 0 and self.step_count % 100 == 0:
                    applied_force = xfrc_target[0:3].detach().cpu().numpy() if hasattr(xfrc_target, 'detach') else xfrc_target[0:3]
                    print(f"[WireAssist] 力が適用されました (Body ID: {self.body_id}): {applied_force}")
            except Exception as e:
                print(f"⚠️ [WireAssist] 力の適用に失敗: {e}")

        return self.env.step(action)

    def reset(self, **kwargs):
        self._force_velocity_command()
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
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""


  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # === 追加: 毎ステップで前進コマンドを強制的に上書きするラッパー ===
  env = WireAssistWrapper(env, pitch_threshold=0.20, tension=300.0, lin_vel_x=0.5, lin_vel_y=0.0, yaw_vel=0.0)
  # =====================================================================
  
  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()