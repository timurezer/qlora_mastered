"""Apply run D's deterministic seed-42 shuffle to the teacher-distillation data."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/base_distill/ds_teacher/train_teacher.json"
OUTPUT = ROOT / "data/base_distill/ds_teacher/shuf42/train_teacher.json"

with SOURCE.open(encoding="utf-8") as f:
    rows = json.load(f)
assert len(rows) == 14090, f"expected 14090 samples, found {len(rows)}"

permutation = list(range(len(rows)))
random.Random(42).shuffle(permutation)
shuffled = [rows[i] for i in permutation]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(shuffled, f, ensure_ascii=False, indent=4)
print(f"Wrote {len(shuffled)} shuffled samples to {OUTPUT}")
