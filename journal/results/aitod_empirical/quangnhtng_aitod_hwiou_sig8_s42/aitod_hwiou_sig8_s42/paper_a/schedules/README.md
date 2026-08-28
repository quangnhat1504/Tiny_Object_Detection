# Train-Only Scale Schedules

These files audit target scales after the exact torchvision detector resize.
They do not contain validation metrics and do not authorize model training.

## Coordinate Contract

For an input of height `H` and width `W`, Faster R-CNN uses
`min_size=640,max_size=800`. The resize factor is

```text
min(640 / min(H,W), 800 / max(H,W))
```

PyTorch interpolation floors the resulting spatial dimensions. Box width and
height are then scaled by the actual integer output-width/input-width and
output-height/input-height ratios. Schedule scale is
`sqrt(resized_box_width * resized_box_height)` in detector-input pixels.

`tests/test_train_scale_schedule.py` checks this calculation against
`GeneralizedRCNNTransform` on square, rectangular, capped, and non-integral
resize cases. The fitter accepts only `split=train` and excludes ignore,
uncertain, crowd, invalid, and out-of-image annotations.

## Current Audit

`aitodv2_train_p10_p90.json` uses only the hash-locked official AI-TOD-v2 train
annotations. Its P10/P90 values are candidate dataset-specific scale bounds;
they are not yet a frozen Paper A run config. No beta or position-weight
endpoint was selected by this audit.

`tinyperson_train_p10_p90.json` uses only the hash-locked official TinyPerson
erased corner-task train annotation
(`tiny_set_train_sw640_sh512_all.json`, SHA-256 `8474f124...`). Its P10/P90
bounds are `7.4328/44.8468 px` detector-input pixels over 32,430 positives;
the distribution is far wider than AI-TOD-v2 (median `15.72 px`, max
`335.22 px`). These are candidate dataset-specific bounds, not a frozen run
config; no beta or position-weight endpoint was selected by this audit. An
independent recomputation reproduced both percentiles and the audit hash.
Reusing the legacy SOD constants remains prohibited.
