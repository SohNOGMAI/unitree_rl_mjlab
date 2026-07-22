"""Run paired nominal/assisted wire traversal trials for the paper table."""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch

# Allow direct execution from the repository root or scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from scripts.play import PlayConfig, run_play  # noqa: E402
from src.wire_eval import (  # noqa: E402
  build_summary_table,
  invalid_trial_summary,
  load_summary_trials,
  write_summary_table,
  write_summary_trials,
)


PRIMITIVE_TO_MODE = {
  "hoist": "ascend",
  "brake": "descend",
  "swing": "gap",
}

DEFAULT_WALKING_CHECKPOINT = (
  "logs/rsl_rl/g1_velocity/2026-06-20_00-41-59/model_10000.pt"
)
DEFAULT_CROUCH_CHECKPOINT = (
  "logs/rsl_rl/g1_crouch/"
  "2026-07-19_11-51-25_gap_planted_no_yaw_s24/model_675.pt"
)

DEFAULT_TIMEOUT_S = {
  # These include approach, posture handover, length-limited winching and the
  # post-traversal stability window.  A universal 20 s timeout cuts Hoist off
  # while it is still in LIFT at the configured 0.05 m/s reel speed.
  "hoist": 45.0,
  "brake": 30.0,
  "swing": 35.0,
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Evaluate Hoist/Brake/Swing with matched nominal and assisted seeds."
    )
  )
  parser.add_argument("task", nargs="?", default="Unitree-G1-Flat")
  parser.add_argument(
    "--primitives",
    nargs="+",
    choices=tuple(PRIMITIVE_TO_MODE),
    default=list(PRIMITIVE_TO_MODE),
  )
  parser.add_argument(
    "--conditions",
    nargs="+",
    choices=("nominal", "assisted"),
    default=["nominal", "assisted"],
  )
  parser.add_argument("--num-trials", type=int, default=10)
  parser.add_argument("--seed-start", type=int, default=0)
  parser.add_argument(
    "--timeout-s",
    type=float,
    default=None,
    help=(
      "Override every scenario timeout. Defaults: Hoist 45 s, Brake 30 s, "
      "Swing 35 s, based on the existing motion-sequence durations."
    ),
  )
  parser.add_argument(
    "--max-attitude-deg",
    type=float,
    default=None,
    help=(
      "Override the existing task fall limit. By default this is 70 deg for "
      "all primitives; 45 deg is used only if no existing limit is available."
    ),
  )
  parser.add_argument("--stability-duration-s", type=float, default=1.0)
  parser.add_argument("--device", default=None)
  parser.add_argument("--checkpoint-file", default=DEFAULT_WALKING_CHECKPOINT)
  parser.add_argument(
    "--crouch-checkpoint-file",
    default=DEFAULT_CROUCH_CHECKPOINT,
  )
  parser.add_argument("--output-dir", default="results/wire_eval")
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Keep completed trial rows whose per-trial CSV still exists.",
  )
  parser.add_argument(
    "--rerun-existing",
    action="store_true",
    help=(
      "Load existing summaries but rerun and replace the selected trial keys."
    ),
  )
  parser.add_argument(
    "--plots",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Generate the three diagnostic PDF plots after aggregation.",
  )
  return parser.parse_args()


def _write_readme(output_dir: Path, args: argparse.Namespace) -> None:
  attitude_text = (
    f"the explicit {args.max_attitude_deg:g} deg override"
    if args.max_attitude_deg is not None
    else "the velocity task's existing 70 deg fall limit"
  )
  timeout_text = (
    f"{args.timeout_s:g} s for every primitive"
    if args.timeout_s is not None
    else "45 s for Hoist, 30 s for Brake, and 35 s for Swing"
  )
  text = f"""# Wire-assisted primitive evaluation

This directory is produced by `scripts/evaluate_wire_primitives.py` on branch
`dev/gap`.  The experiment compares the same 29-DoF walking policy, terrain,
reset distribution and seed with two conditions:

- **Nominal:** continuous +x walking. Cable force, assist state machine,
  learned crouch handover and fixed suspension postures are bypassed. Both
  policy checkpoints are loaded, but the zero blend returns exactly the same
  walking-actor action used in Assisted before handover.
- **Assisted:** the existing `WireAssistWrapper` controller is used unchanged,
  including its length/tension controller and primitive-specific posture logic.

Primitive naming is: `ascend -> hoist`, `descend -> brake`, `gap -> swing`.
Swing--Hoist and Swing--Brake are not part of this evaluation.

## Reproduce

```bash
python scripts/evaluate_wire_primitives.py {args.task} \\
  --checkpoint-file {args.checkpoint_file} \\
  --crouch-checkpoint-file {args.crouch_checkpoint_file} \\
  --device cuda:0 \\
  --num-trials 10
```

Quick smoke test:

```bash
python scripts/evaluate_wire_primitives.py {args.task} \\
  --checkpoint-file {args.checkpoint_file} \\
  --crouch-checkpoint-file {args.crouch_checkpoint_file} \\
  --device cuda:0 --num-trials 1
```

Regenerate plots without rerunning simulation:

```bash
python scripts/plot_wire_eval.py --results-dir {args.output_dir}
```

`--num-trials N --seed-start S` uses seeds `S` through `S + N - 1`.
The defaults are `S=0`, `N=10`. `--primitives` and `--conditions` can select a
subset; `--resume` skips summarized trials and `--rerun-existing` replaces the
selected keys while preserving the other rows.

## Metrics and protocol

The logged assistance ratio is

`rho_z = T * u_z / (m * g)`,

where `T` is the actually applied cable tension, `u_z` is the vertical
component of the unit vector from the back attachment to the anchor, `m` is
`MjModel.body_subtreemass` for the robot root, and `g` is the magnitude of
`MjModel.opt.gravity`.  Thus the computation uses the model's actual mass and
gravity rather than copied constants.  Nominal `rho_z` and tension are exactly
zero.  Cable attachment and anchor positions are read from the active
`WireAssistWrapper` instance.

A trial succeeds after reaching the geometry-derived target region while at
least one foot has actual terrain contact, and remaining below {attitude_text}
in absolute torso roll/pitch for {args.stability_duration_s:g} s. Controller
phase names are logged but are not success conditions. `completion_time_s`
includes that stability window. Timeout is {timeout_text}. A torso attitude
above 70 deg or a
floating-base height below the support floor plus 0.30 m is classified as
`fall`; an attitude between the evaluation and fall thresholds is
`excessive_attitude`.

Target regions are derived from the existing obstacle/gap box bounds, not from
new hard-coded world coordinates.  `traversal_distance_or_height` is maximum
base height gain for Hoist, maximum base height loss for Brake, and maximum +x
base displacement for Swing.  `final_position_error` is the Euclidean shortfall
from the target-region boundary (zero once inside it).

## Files

- `trials/<primitive>_<condition>_seedNNN.csv`: per-step measurements.
- `summary_trials.csv`: one row per trial.
- `summary_table.csv`: Table-II-ready paired aggregate.  Time, `rho_z`, and
  attitude statistics use sample standard deviation (`ddof=1`; 0 for one
  trial). Completion time uses successful Assisted trials only; peak `rho_z`
  and maximum attitude use every finite Assisted trial, including failures.
- `plots/<primitive>_assisted_timeseries.pdf`: the lowest available Assisted
  seed for each primitive, with controller-state transitions marked.
- `errors.log`: Python traceback only when a trial cannot be constructed/run.

## Known unavailable measurement

The existing `feet_ground_contact` sensor supplies the landing-contact check.
In evaluation its primary foot selection is unchanged, but its secondary
filter accepts any opposing geom so contacts with the step and gap platforms
are included as well as the flat terrain.
MJWarp's data bridge used by this project does not expose a dependable named
geom-pair contact table to this logger. Therefore `gap_contact` detects only
the unambiguous gross event in which the floating base enters the gap below the
platform top.  Exact limb-to-edge contact counts are **not inferred** and are
not included as a numeric metric.  All other requested values are read directly
from simulation/controller state; if a trial throws or produces non-finite
state, its metrics are `NaN` and the reason is `invalid_simulation`.
"""
  (output_dir / "README.md").write_text(text, encoding="utf-8")


def _existing_rows(output_dir: Path, resume: bool) -> list[dict[str, str]]:
  path = output_dir / "summary_trials.csv"
  if not resume or not path.is_file():
    return []
  return load_summary_trials(path)


def main() -> None:
  args = _parse_args()
  if args.num_trials <= 0:
    raise ValueError("--num-trials must be positive")
  if (
    (args.timeout_s is not None and args.timeout_s <= 0.0)
    or args.stability_duration_s <= 0.0
  ):
    raise ValueError("timeout and stability duration must be positive")
  if (
    args.max_attitude_deg is not None
    and not 0.0 < args.max_attitude_deg <= 180.0
  ):
    raise ValueError("--max-attitude-deg must be in the interval (0, 180]")

  output_dir = Path(args.output_dir).expanduser().resolve()
  trials_dir = output_dir / "trials"
  trials_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "plots").mkdir(parents=True, exist_ok=True)
  _write_readme(output_dir, args)

  summaries: list[dict] = list(
    _existing_rows(output_dir, args.resume or args.rerun_existing)
  )
  completed = set() if args.rerun_existing else {
    (row["primitive"], row["condition"], int(row["seed"]))
    for row in summaries
    if (trials_dir / f"{row['primitive']}_{row['condition']}_seed{int(row['seed']):03d}.csv").is_file()
  }
  errors_path = output_dir / "errors.log"

  for primitive in args.primitives:
    mode = PRIMITIVE_TO_MODE[primitive]
    for seed in range(args.seed_start, args.seed_start + args.num_trials):
      # Pair conditions at each seed so reset randomness is directly matched.
      for condition in args.conditions:
        key = (primitive, condition, seed)
        if key in completed:
          print(f"[BATCH] skip completed {primitive}/{condition}/seed{seed:03d}")
          continue
        trial_csv = trials_dir / f"{primitive}_{condition}_seed{seed:03d}.csv"
        print(
          f"[BATCH] start {primitive}/{condition}/seed{seed:03d}",
          flush=True,
        )
        cfg = PlayConfig(
          traversal_mode=mode,
          checkpoint_file=args.checkpoint_file,
          crouch_checkpoint_file=args.crouch_checkpoint_file,
          device=args.device,
          num_envs=1,
          video=False,
          no_terminations=True,
          evaluation_condition=condition,
          evaluation_output_csv=str(trial_csv),
          evaluation_seed=seed,
          evaluation_timeout_s=(
            args.timeout_s
            if args.timeout_s is not None
            else DEFAULT_TIMEOUT_S[primitive]
          ),
          evaluation_max_attitude_deg=args.max_attitude_deg,
          evaluation_stability_duration_s=args.stability_duration_s,
        )
        try:
          summary = run_play(args.task, cfg)
        except KeyboardInterrupt:
          raise
        except Exception:
          summary = invalid_trial_summary(
            primitive,
            condition,
            seed,
            trial_csv,
          )
          with errors_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {primitive}/{condition}/seed{seed:03d} ===\n")
            traceback.print_exc(file=stream)
          print(
            f"[BATCH][ERROR] {primitive}/{condition}/seed{seed:03d}; "
            f"see {errors_path}",
            flush=True,
          )
        summaries = [
          row for row in summaries
          if (row["primitive"], row["condition"], int(row["seed"])) != key
        ]
        summaries.append(summary)
        summaries.sort(
          key=lambda row: (
            ("hoist", "brake", "swing").index(row["primitive"]),
            int(row["seed"]),
            ("nominal", "assisted").index(row["condition"]),
          )
        )
        write_summary_trials(summaries, output_dir / "summary_trials.csv")
        write_summary_table(
          build_summary_table(summaries),
          output_dir / "summary_table.csv",
        )
        gc.collect()
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

  if args.plots:
    from scripts.plot_wire_eval import create_plots

    create_plots(output_dir)

  table_path = output_dir / "summary_table.csv"
  print(f"[BATCH] finished; summary: {table_path}", flush=True)
  print(table_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
  main()
