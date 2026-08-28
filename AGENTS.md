# Repository Guidelines

## Project Structure & Module Organization

- `common/` contains the Faster R-CNN model, dataset pipeline, metrics, and
  training utilities shared by experiments.
- `scripts/` holds runnable training, evaluation, smoke, and audit scripts.
- `paper_a/` is the only submission-facing SA-ALW workspace. Its canonical
  method, tests, split/evaluator contracts, ledgers, and experiment reports
  live here. Treat `paper/` as historical diagnostics, not paper evidence.
- `data/`, `eda/`, `runs/`, and `.runtime/` contain datasets, exploration,
  generated run outputs, and local Kaggle orchestration state respectively.
- `wiki/` records project decisions and experiment context; do not edit raw
  sources under `raw/`.

## Build, Test, and Development Commands

Use the CUDA environment for model-dependent checks:

```powershell
.\.venv-cuda\Scripts\python.exe -m unittest discover -s paper_a\tests -p "test_*.py"
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_phase0.py
.\.venv-cuda\Scripts\python.exe -m py_compile common\model.py scripts\train_frcnn_metric.py
```

The first command runs Paper A unit and protocol tests; the second validates
its ledgers; the third is a quick syntax check for edited training code. Run
EDA with `.venv\Scripts\python.exe eda\eda_tinyperson.py` when its optional
dependencies are installed. Kaggle runs use one experiment per invocation,
for example `python scripts\kaggle_run.py --run cbl_iterative_train_fair20 --data-root <path>`.

## Coding Style & Naming Conventions

Use Python 4-space indentation, type hints for new public helpers, descriptive
`snake_case` functions/files, `PascalCase` classes, and `UPPER_CASE` constants.
Keep experiment names explicit (for example, `pc_mr_moc_*` or
`wp03_*_s123`). Avoid broad refactors in `common/`; changes there affect many
experiments. No formatter or linter is configured, so preserve nearby style.

## Testing Guidelines

Add `unittest` tests named `test_<behavior>.py`; place Paper A tests in
`paper_a/tests/` and experiment-specific contract tests in `scripts/`.
Cover geometry, artifact/schema, and reload behavior before attempting GPU
training. Treat local multi-epoch scores as diagnostics only. Kaggle results
are accepted only after downloaded artifacts, logs, and independent reload
checks pass.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects such as `Add detached RPN IoU
quality experiment`. Keep commits scoped to one method, audit, or documentation
change. PRs should state the hypothesis, affected paths, test commands and
results, compute/data impact, and artifact locations. Include screenshots only
for visual EDA or manuscript changes. Never include credentials, private data,
or final-test outputs; Paper A pushes require separate pre-run and post-run
reports.
