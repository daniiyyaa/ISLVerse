"""Train an English sentence -> ISL gloss classifier (TF-IDF + logistic regression).

The model picks one of the known glosses for a sentence, so it handles new
phrasings of known intents, not brand-new meanings. For evaluation it holds out
one paraphrase per intent for val and one for test (variant 1 always stays in
train). The intent-grouped train/val/test.csv cannot be used here: their test
intents never appear in training, so a classifier cannot predict them.

Run from the repo root:  python data/train.py
"""
import argparse, random, sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

sys.path.insert(0, "src")
from isl_preprocess import load_lookup, normalize, to_gloss

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--C", type=float, default=20.0)
ap.add_argument("--threshold", type=float, default=0.3,
                help="below this probability, return None instead of a guess")
ap.add_argument("--out", default="models/isl_gloss_clf.joblib")
args = ap.parse_args()

df = pd.read_csv("data/isl_dataset_full.csv", keep_default_na=False)
df["text"] = df["English Sentence"].map(normalize)

# hold out one paraphrase per intent for val and one for test
rng = random.Random(args.seed)
split = {}
for _, grp in df.groupby("Intent ID"):
    ids = sorted(grp.loc[grp["Variant No."] != 1, "ID"]); rng.shuffle(ids)
    split[ids[0]], split[ids[1]] = "test", "val"
df["Split"] = df["ID"].map(split).fillna("train")
# a sentence that also exists in train is not a real held-out example
train_texts = set(df.loc[df.Split == "train", "text"])
df.loc[(df.Split != "train") & df.text.isin(train_texts), "Split"] = "train"


def build():
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
    return make_pipeline(feats, LogisticRegression(C=args.C, max_iter=3000))


def evaluate(model, part, name):
    proba = model.predict_proba(part["text"])
    conf = proba.max(axis=1)
    pred = model.classes_[proba.argmax(axis=1)]
    correct = pred == part["ISL Gloss"].values
    kept = conf >= args.threshold
    print(f"{name:5} n={len(part):4}  accuracy={correct.mean():.3f}  "
          f"above-threshold: {kept.mean():.0%} kept, accuracy={correct[kept].mean():.3f}")
    return pred, correct


train = df[df.Split == "train"]
print(f"train={len(train)}  val={(df.Split == 'val').sum()}  test={(df.Split == 'test').sum()}  "
      f"glosses={train['ISL Gloss'].nunique()}")
model = build().fit(train["text"], train["ISL Gloss"])
evaluate(model, df[df.Split == "val"], "val")
test = df[df.Split == "test"]
pred, correct = evaluate(model, test, "test")

print("\nSample test errors:")
for (_, row), p in list(zip(test[~correct].iterrows(), pred[~correct]))[:10]:
    print(f"  {row.text!r:45} want {row['ISL Gloss']!r:28} got {p!r}")

# final model: refit on every row and save
final = build().fit(df["text"], df["ISL Gloss"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": final, "threshold": args.threshold}, args.out)
print(f"\nSaved final model (all {len(df)} rows) to {args.out}")


def predict(sentence, bundle, lookup):
    """Exact match / NAME / AGE slots first, then the classifier."""
    gloss = to_gloss(sentence, lookup)
    if gloss is not None:
        return gloss
    proba = bundle["model"].predict_proba([normalize(sentence)])[0]
    return bundle["model"].classes_[proba.argmax()] if proba.max() >= bundle["threshold"] else None


lookup = load_lookup()
bundle = joblib.load(args.out)
print("\nDemo:")
for s in ["My name is Ben", "I am 21 years old", "are you free this evening",
          "thank you so much", "i feel really hungry", "the quantum flux capacitor"]:
    print(f"  {s!r:32} -> {predict(s, bundle, lookup)}")
