"""Create diagnostic time-series PDFs from wire evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_COLORS = {
  "APPROACH": "#f2f2f2",
  "SETTLE": "#fff2cc",
  "LIFT": "#d9eaf7",
  "HOLD": "#d9ead3",
  "LOWER": "#eadcf8",
  "ASSIST": "#fce5cd",
  "RELEASE": "#f4cccc",
  "DONE": "#d0e0e3",
}


def _number(row: dict[str, str], field: str) -> float:
  try:
    return float(row[field])
  except (KeyError, TypeError, ValueError):
    return math.nan


def _shade_states(axes, times: np.ndarray, states: list[str]) -> None:
  if times.size == 0:
    return
  starts = [0]
  for index in range(1, len(states)):
    if states[index] != states[index - 1]:
      starts.append(index)
  starts.append(len(states))
  for begin, end in zip(starts[:-1], starts[1:]):
    state = states[begin]
    x0 = times[begin]
    x1 = times[end - 1] if end < len(times) else times[-1]
    if x1 <= x0 and begin + 1 < len(times):
      x1 = times[begin + 1]
    for axis in axes:
      axis.axvspan(
        x0,
        x1,
        color=STATE_COLORS.get(state, "#eeeeee"),
        alpha=0.22,
        linewidth=0,
      )
      if begin > 0:
        axis.axvline(x0, color="0.55", linewidth=0.7, linestyle="--")
    axes[0].text(
      x0,
      1.02,
      state,
      transform=axes[0].get_xaxis_transform(),
      fontsize=7,
      rotation=35,
      ha="left",
      va="bottom",
    )


def _plot_one(csv_path: Path, output_path: Path, primitive: str) -> None:
  with csv_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  if not rows:
    print(f"[PLOT][WARN] no samples in {csv_path}")
    return

  times = np.asarray([_number(row, "time_s") for row in rows])
  rho = np.asarray([_number(row, "rho_z") for row in rows])
  roll = np.asarray([_number(row, "torso_roll_deg") for row in rows])
  pitch = np.asarray([_number(row, "torso_pitch_deg") for row in rows])
  length = np.asarray([_number(row, "wire_length_m") for row in rows])
  target = np.asarray([_number(row, "target_wire_length_m") for row in rows])
  states = [row.get("state_machine_state", "UNKNOWN") for row in rows]

  fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.2), sharex=True)
  _shade_states(axes, times, states)
  axes[0].plot(times, rho, color="#0072b2", linewidth=1.4)
  axes[0].axhline(1.0, color="0.35", linestyle=":", linewidth=0.9)
  axes[0].set_ylabel(r"$\rho_z$")

  axes[1].plot(times, roll, label="roll", color="#d55e00", linewidth=1.2)
  axes[1].plot(times, pitch, label="pitch", color="#009e73", linewidth=1.2)
  axes[1].axhline(45.0, color="0.4", linestyle=":", linewidth=0.8)
  axes[1].axhline(-45.0, color="0.4", linestyle=":", linewidth=0.8)
  axes[1].set_ylabel("Torso attitude [deg]")
  axes[1].legend(loc="best", frameon=False, ncols=2)

  axes[2].plot(times, length, label="measured", color="#cc79a7", linewidth=1.3)
  if np.isfinite(target).any():
    axes[2].plot(
      times,
      target,
      label="commanded",
      color="#000000",
      linewidth=1.0,
      linestyle="--",
    )
  axes[2].set_ylabel("Wire length [m]")
  axes[2].set_xlabel("Time [s]")
  axes[2].legend(loc="best", frameon=False)

  for axis in axes:
    axis.grid(True, alpha=0.25)
  fig.suptitle(f"{primitive.capitalize()} assisted trial: {csv_path.stem}")
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)
  print(f"[PLOT] wrote {output_path}")


def create_plots(results_dir: str | Path) -> None:
  results = Path(results_dir)
  trial_dir = results / "trials"
  plot_dir = results / "plots"
  for primitive in ("hoist", "brake", "swing"):
    candidates = sorted(trial_dir.glob(f"{primitive}_assisted_seed*.csv"))
    if not candidates:
      print(f"[PLOT][WARN] no Assisted CSV for {primitive}")
      continue
    _plot_one(
      candidates[0],
      plot_dir / f"{primitive}_assisted_timeseries.pdf",
      primitive,
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--results-dir", default="results/wire_eval")
  args = parser.parse_args()
  create_plots(Path(args.results_dir).expanduser().resolve())


if __name__ == "__main__":
  main()
