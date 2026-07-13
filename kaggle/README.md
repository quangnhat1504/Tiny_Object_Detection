# Kaggle runner

This repo runs on Kaggle by cloning the GitHub repository and executing the existing training scripts. Do not rewrite the training loop inside a notebook.

`script/kaggle_run.py` is a thin dispatcher. It runs **one experiment per invocation**, so use separate Kaggle notebooks/accounts for separate experiments.

## Minimal Kaggle notebook cells

```bash
!git clone https://github.com/quangnhat1504/Tiny_Object_Detection.git /kaggle/working/tiny-object-detection
%cd /kaggle/working/tiny-object-detection
!pip install -q -r requirements-kaggle.txt
```

## Recommended split across Kaggle accounts

Account/notebook 1, priority localization baseline:

```bash
!python scripts/kaggle_run.py --run smooth_l1_ap75 --data-root /kaggle/input/sod-tinypeopleinsea
```

Account/notebook 2, reduced oversampling:

```bash
!python scripts/kaggle_run.py --run os1 --data-root /kaggle/input/sod-tinypeopleinsea
```

Account/notebook 3, light copy-paste:

```bash
!python scripts/kaggle_run.py --run cp_light --data-root /kaggle/input/sod-tinypeopleinsea
```

The remaining accounts can duplicate the most promising run with another seed/tag after we add it, or hold for resume/retry if a Kaggle session times out.

## Analysis only

After a completed checkpoint exists in the same Kaggle working directory:

```bash
!python scripts/kaggle_run.py --run smooth_l1_ap75 --skip-train --data-root /kaggle/input/sod-tinypeopleinsea
```

## Notes

- The runner sets `CPV_DATA_ROOT` for Kaggle and clears `PYTORCH_CUDA_ALLOC_CONF` because platform support differs.
- Start with one T4 per notebook for reproducibility. Two T4s require DDP or a separate parallel-run strategy.
- Kaggle outputs are written under `/kaggle/working/tiny-object-detection/runs`.
- To resume across Kaggle sessions, persist `runs/.../last.pt` as a Kaggle Dataset or upload it back before running with `--resume`.
