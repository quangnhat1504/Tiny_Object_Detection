# Split Status

`legacy_split_manifest.csv` inventories the current Roboflow derivative. It is
not a frozen Paper A split. `split_audit.json` records why it fails G2.

The required submission `split_manifest.csv` does not exist yet. It must come
from an official public split or a source/sequence-disjoint manifest frozen
before tuning. Repartitioning the current 654 already exposed sources cannot
create a fresh final test.

