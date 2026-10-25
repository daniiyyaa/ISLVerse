"""Create train/val/test splits grouped by Intent ID (no paraphrase leakage).

Also keeps together any intents that share an identical English sentence
(e.g. the two "what are you doing" rows), so one sentence never appears
in two splits with possibly different glosses.

Splits are stratified by Category (~80/10/10) and reproducible via --seed.
Run from the repo root:  python src/make_splits.py
"""
import argparse, random
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--val", type=float, default=0.10)
ap.add_argument("--test", type=float, default=0.10)
args = ap.parse_args()

data = Path("data")
df = pd.read_csv(data / "isl_dataset_full.csv", keep_default_na=False)

# union-find: intents sharing an English sentence form one unit
parent = {i: i for i in df["Intent ID"].unique()}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for _, grp in df.groupby("English Sentence"):
    ids = grp["Intent ID"].unique()
    for other in ids[1:]:
        parent[find(other)] = find(ids[0])

df["_unit"] = df["Intent ID"].map(find)
unit_cat = df.groupby("_unit")["Category"].first()

rng = random.Random(args.seed)
split_of = {}
for cat, units in unit_cat.groupby(unit_cat):
    units = sorted(units.index); rng.shuffle(units)
    n = len(units)
    n_test = max(1, round(n * args.test)); n_val = max(1, round(n * args.val))
    for u in units[:n_test]: split_of[u] = "test"
    for u in units[n_test:n_test + n_val]: split_of[u] = "val"
    for u in units[n_test + n_val:]: split_of[u] = "train"

df["Split"] = df["_unit"].map(split_of)
for name in ("train", "val", "test"):
    part = df[df["Split"] == name].drop(columns=["_unit", "Split"])
    part.to_csv(data / f"{name}.csv", index=False, encoding="utf-8")
    print(f"{name:5} rows={len(part):5}  intents={part['Intent ID'].nunique()}")

# sanity checks
tr, va, te = [set(df[df.Split == s]["Intent ID"]) for s in ("train", "val", "test")]
assert not (tr & va or tr & te or va & te), "intent leakage"
sent = df.groupby("English Sentence")["Split"].nunique()
assert (sent == 1).all(), "sentence appears in multiple splits"
print("OK: no Intent ID or sentence shared across splits")
