# TinyPerson Evaluator Runtime Compatibility

The official AP evaluator is loaded from the source blob pinned at commit
`bf6b83aa9a149ae15087eed4e9a7283f5cc67603`. Before execution, the wrapper
normalizes Windows line endings and verifies SHA-256
`222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557`.

The evaluator predates NumPy 2. Its five `np.linspace` calls pass a NumPy float
as the sample count, and its accumulator uses the removed `np.float` alias.
After hash verification, the runtime loader applies only these compatibility
changes:

- cast the two locked `np.round(...) + 1` expression forms to `int`;
- map `np.float` to Python `float` when the alias is absent.

The IoU thresholds, recall thresholds, area ranges, `maxDets`, matching logic,
uncertain handling, and IOD ignore matching are unchanged. The fixture test
places a higher-scoring detection fully inside an uncertain region before a
true positive; AP remains `1.0` only when the official IOD ignore path works.
