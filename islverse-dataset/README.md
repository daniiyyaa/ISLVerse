# ISLverse Dataset

English sentence → Indian Sign Language (ISL) gloss pairs for the **ISLverse** project, which converts live speech transcription into sign language using gesture recognition and a 3D Blender avatar.

> **Review status:** the glosses and paraphrases are *proposals*. They have **not yet been verified by an ISL expert or native signer**. Have them reviewed before using this data for final model training or publishing results.

## Contents

| Path | Description |
|---|---|
| `data/isl_dataset_full.csv` | Full dataset (2,226 rows, 318 intents, 14 categories) |
| `data/train.csv`, `val.csv`, `test.csv` | Splits grouped by `Intent ID` (~80/10/10, stratified by category) |
| `src/make_splits.py` | Regenerates the splits (`python src/make_splits.py --seed 42`) |
| `src/isl_preprocess.py` | Sentence → gloss lookup with NAME and AGE slot filling |
| `data/train.py` | TF-IDF + logistic regression classifier that picks one of the known glosses |
| `data/train_mt5.py` | Fine-tunes Flan-T5 (or mT5) to generate glosses for unseen sentences |
| `source/*.xlsx` | Excel version with the Category Summary and Notes sheets |

## Columns

| Column | Meaning |
|---|---|
| `ID` | Row number |
| `Intent ID` | Groups all phrasings that share one meaning and one gloss |
| `Category` | One of 14 categories (Greetings, Questions, Feelings, Safety and Emergency, ...) |
| `English Sentence` | Lowercase, no apostrophes or question marks |
| `ISL Gloss` | Uppercase gloss |
| `Variant No.` | 1 = base sentence, 2–7 = generated paraphrases |
| `Source` | `Original cleaned dataset`, `Proposed additional examples`, `Generated new intent` or `Generated variation` |
| `Placeholder` | `NAME` or `AGE` if the row contains a slot (`xxxxxxxx` or `(age)`), else empty |

## Split sizes

| Split | Rows | Intents |
|---|---|---|
| train | 1,771 | 253 |
| val | 224 | 32 |
| test | 231 | 33 |

**Always split by `Intent ID`, never by row.** Paraphrases of one intent in both train and test would inflate accuracy. Intents that share an identical English sentence are also kept in the same split.

## Names and ages

Train on the placeholder forms (`my name is xxxxxxxx` → `MY NAME XXXXXXXX`, `i am (age)` → `I (AGE)`). At run time `isl_preprocess.py` detects the slot and fills it in:

```python
from isl_preprocess import load_lookup, to_gloss
lookup = load_lookup("data/isl_dataset_full.csv")
to_gloss("My name is Ben", lookup)     # MY NAME B-E-N   (fingerspelled)
to_gloss("I am 21 years old", lookup)  # I 21
```

Order of resolution: exact dataset match → name pattern → age pattern → `None`. "i am ben" is deliberately *not* treated as a name because it is indistinguishable from "i am tired".

## Known issues

- "what are you doing" has two glosses (`WHAT YOU DO` / `YOU DO WHAT`) and "you are welcome" has two (`YOU WELCOME` / `WELCOME`). The lookup uses the first one.
- `I DONOT AGREE` and `I DONT AGREE` are inconsistent spellings of the same gloss.
- WH-word position differs: original rows put it first (`WHAT YOU WANT`), proposed/new rows put it last (`YOU GO WHERE`). Pick one convention before final training.
- Some glosses keep English helper words (e.g. `I AM SO SORRY TO HEAR THAT`).
- General Communication (57 intents) is much larger than other categories (20–21).
- Some generated paraphrases add an intensifier ("so", "very") that the gloss does not carry.

## Setup

```bash
pip install -r requirements.txt
python src/make_splits.py
python src/isl_preprocess.py
```

All commands run from this folder.

## Training the gloss generator

```bash
python data/train_mt5.py                        # train on train, pick best epoch on val, score on test
python data/train_mt5.py --final --epochs 4     # train on train+val for the final model
```

Models are saved under `models/` (not committed; flan-t5-base is ~1 GB). Glosses are trained in lowercase and uppercased after generation so the model can copy unseen words from the input.

Results on the held-out test intents (231 sentences, 33 intents never seen in training):

| Model | Exact match | Word error rate |
|---|---|---|
| flan-t5-base, train+val, 4 epochs | 0.251 | 0.556 |

The main limit is the dataset: only 24% of test sentences have a gloss made entirely of words seen in training. More distinct intents will help more than more paraphrases.

## License

Not yet chosen. Add a `LICENSE` file before making the repository public (CC BY 4.0 is common for datasets, MIT for code).
