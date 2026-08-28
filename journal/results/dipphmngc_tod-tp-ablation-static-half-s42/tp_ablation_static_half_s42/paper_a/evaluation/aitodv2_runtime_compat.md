# AI-TOD-v2 Evaluator Runtime Compatibility

The semantic evaluator is pinned to `cocoapi-aitod` commit
`44a230ae5197cb89bf9e5e62f313cac3ad30c7af`. Its `cocoeval.py` is used without
modification and is hash-locked in `official_evaluator_lock.json`.

The historical package needs three build/runtime compatibility shims on the
current Python, Cython, and NumPy stack:

1. Remove the obsolete relative C source directive from `_mask.pyx`; setup.py
   already compiles the same `maskApi.c` source.
2. Replace the removed Cython-facing `np.NPY_OWNDATA` symbol with its NumPy flag
   value `4` in `_mask.pyx`.
3. Define `np.float = float` before importing the historical evaluator because
   NumPy 1.24 removed that alias.

These shims do not change evaluator thresholds, matching, accumulation, or
summary logic. The perfect-box fixture in
`tests/test_aitod_official_evaluator.py` must pass locally and again inside the
Kaggle runtime before any benchmark job is released.
