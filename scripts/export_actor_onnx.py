"""Export an rsl_rl MLP actor checkpoint to a self-contained ONNX file.

This utility is intentionally small and deterministic.  It is used for the
real-robot deployment, where the C++ controller cannot load a PyTorch `.pt`
checkpoint directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class ExportedActor(nn.Module):
  """Inference-only actor matching rsl_rl's exported MLP model."""

  def __init__(self, state: dict[str, torch.Tensor]) -> None:
    super().__init__()
    layer_ids = sorted(
      {
        int(key.split(".")[1])
        for key in state
        if key.startswith("mlp.") and key.endswith(".weight")
      }
    )
    if not layer_ids:
      raise ValueError("Checkpoint contains no mlp.*.weight tensors")

    layers: list[nn.Module] = []
    for index, layer_id in enumerate(layer_ids):
      weight = state[f"mlp.{layer_id}.weight"]
      bias = state[f"mlp.{layer_id}.bias"]
      linear = nn.Linear(weight.shape[1], weight.shape[0])
      linear.weight.data.copy_(weight)
      linear.bias.data.copy_(bias)
      layers.append(linear)
      if index != len(layer_ids) - 1:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)
    self.register_buffer("mean", state["obs_normalizer._mean"].clone())
    self.register_buffer("std", state["obs_normalizer._std"].clone())

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    normalized = (obs - self.mean) / (self.std + 1.0e-2)
    return self.mlp(normalized)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()

  checkpoint = torch.load(
    args.checkpoint, map_location="cpu", weights_only=False
  )
  state = checkpoint.get("actor_state_dict")
  if state is None:
    raise KeyError("Checkpoint has no actor_state_dict")

  actor = ExportedActor(state).eval()
  obs_dim = int(state["obs_normalizer._mean"].shape[-1])
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    actor,
    torch.zeros(1, obs_dim),
    args.output,
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  print(
    f"Exported iteration {checkpoint.get('iter', 'unknown')} "
    f"to {args.output}"
  )


if __name__ == "__main__":
  main()
