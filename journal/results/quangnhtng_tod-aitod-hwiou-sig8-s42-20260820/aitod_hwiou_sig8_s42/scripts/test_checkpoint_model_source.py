"""Verify best-checkpoint state selection follows the evaluated model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.train_utils import ModelEMA
from scripts.train_frcnn_metric import _select_evaluation_model


def main() -> None:
    model = nn.Linear(2, 1)
    ema = ModelEMA(model)
    shadow_before = {
        name: value.clone()
        for name, value in ema.state_dict().items()
    }
    with torch.no_grad():
        model.weight.add_(10.0)
        model.bias.add_(10.0)

    eval_model, source = _select_evaluation_model(model, ema)
    assert source == "ema"
    for name, value in eval_model.state_dict().items():
        assert torch.equal(value, shadow_before[name])
        assert not torch.equal(value, model.state_dict()[name])

    raw_model, raw_source = _select_evaluation_model(model, None)
    assert raw_source == "raw"
    assert raw_model is model
    print("checkpoint model-source selection PASSED")


if __name__ == "__main__":
    main()
