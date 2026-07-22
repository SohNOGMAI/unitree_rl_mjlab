# Wire-assisted primitive evaluation

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
python scripts/evaluate_wire_primitives.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-06-20_00-41-59/model_10000.pt \
  --crouch-checkpoint-file logs/rsl_rl/g1_crouch/2026-07-19_11-51-25_gap_planted_no_yaw_s24/model_675.pt \
  --device cuda:0 \
  --num-trials 10
```

Quick smoke test:

```bash
python scripts/evaluate_wire_primitives.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-06-20_00-41-59/model_10000.pt \
  --crouch-checkpoint-file logs/rsl_rl/g1_crouch/2026-07-19_11-51-25_gap_planted_no_yaw_s24/model_675.pt \
  --device cuda:0 --num-trials 1
```

Regenerate plots without rerunning simulation:

```bash
python scripts/plot_wire_eval.py --results-dir results/wire_eval
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
least one foot has actual terrain contact, and remaining below the velocity task's existing 70 deg fall limit
in absolute torso roll/pitch for 1 s. Controller
phase names are logged but are not success conditions. `completion_time_s`
includes that stability window. Timeout is 45 s for Hoist, 30 s for Brake, and 35 s for Swing. A torso attitude
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
