"""Fine-tune a seq2seq model (Flan-T5 by default, mT5 also works) to translate
English sentences into ISL gloss.

Unlike data/train.py (which picks one of the known glosses), this model
generates the gloss word by word, so it can attempt sentences whose meaning is
not in the dataset. It trains on data/train.csv and is scored on val/test,
whose intents never appear in training.

Glosses are trained in lowercase and uppercased after generation: lowercase
English tokenizes into fewer, familiar pieces, so the model can copy words it
never saw as a gloss straight from the input sentence.

Run from the repo root:  python data/train_mt5.py
Faster smoke test:       python data/train_mt5.py --epochs 1
Old mT5 setup:           python data/train_mt5.py --model google/mt5-small --lr 1e-3 --out models/mt5-isl
Final model (train+val): python data/train_mt5.py --final --epochs <best epoch from a normal run>
"""
import argparse, os, random, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# transformers' threaded weight loader crashes (access violation) on Python 3.15 / Windows
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
from transformers.optimization import Adafactor

logging.set_verbosity_error()

sys.path.insert(0, "src")
from isl_preprocess import load_lookup, normalize, to_gloss

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/flan-t5-base")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--eval-every", type=int, default=2, help="epochs between val checks")
ap.add_argument("--final", action="store_true",
                help="train on train+val and save the last epoch (no val selection)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="models/flan-t5-isl")
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
# T5 overflows in fp16; bf16 is safe on RTX 30xx/40xx
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
PREFIX = "translate English to ISL gloss: "

# sentencepiece has no Python 3.15 wheel; mT5 keeps its fast tokenizer under onnx/
tok_kw = {"subfolder": "onnx"} if "mt5" in args.model and not Path(args.model).exists() else {}
tok = AutoTokenizer.from_pretrained(args.model, **tok_kw)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)


def load(split):
    df = pd.read_csv(f"data/{split}.csv", keep_default_na=False)
    return list(zip(df["English Sentence"].map(normalize), df["ISL Gloss"]))


train, val, test = load("train"), load("val"), load("test")
if args.final:
    train = train + val
print(f"model={args.model}  device={device}  train={len(train)}  val={len(val)}  test={len(test)}")


def collate(batch):
    src, tgt = zip(*batch)
    enc = tok([PREFIX + s for s in src], padding=True, truncation=True, max_length=64, return_tensors="pt")
    labels = tok([t.lower() for t in tgt], padding=True, truncation=True, max_length=48,
                 return_tensors="pt").input_ids
    labels[labels == tok.pad_token_id] = -100
    enc["labels"] = labels
    return enc


@torch.no_grad()
def generate(sentences, m=None, bs=64):
    m = m or model
    m.eval()
    out = []
    for i in range(0, len(sentences), bs):
        enc = tok([PREFIX + s for s in sentences[i:i + bs]], padding=True, return_tensors="pt").to(device)
        with torch.autocast(**amp):
            ids = m.generate(**enc, max_new_tokens=32, num_beams=4)
        out += [" ".join(s.split()).upper() for s in tok.batch_decode(ids, skip_special_tokens=True)]
    return out


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def evaluate(pairs, name, m=None):
    src, ref = zip(*pairs)
    hyp = generate(list(src), m)
    exact = sum(a == b for a, b in zip(ref, hyp)) / len(ref)
    w = sum(wer(a, b) for a, b in zip(ref, hyp)) / len(ref)
    print(f"  {name:5} exact match={exact:.3f}  word error rate={w:.3f}")
    return exact, w, hyp


loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False)
best, best_epoch = (-1.0, float("inf")), 0
Path(args.out).mkdir(parents=True, exist_ok=True)

for epoch in range(1, args.epochs + 1):
    model.train()
    t0, total = time.time(), 0.0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(**amp):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        total += loss.item()
    print(f"epoch {epoch:3}  loss={total / len(loader):.4f}  ({time.time() - t0:.0f}s)")

    if args.final:
        continue
    if epoch % args.eval_every == 0 or epoch == args.epochs:
        exact, w, _ = evaluate(val, "val")
        if (exact, -w) > (best[0], -best[1]):      # higher exact match, then lower WER
            best, best_epoch = (exact, w), epoch
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            print(f"  saved best model to {args.out}")

if args.final:
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"\nSaved final model (train+val, {args.epochs} epochs) to {args.out}")
else:
    print(f"\nBest val (epoch {best_epoch}): exact match={best[0]:.3f}  word error rate={best[1]:.3f}")
best_model = AutoModelForSeq2SeqLM.from_pretrained(args.out).to(device)
_, _, hyp = evaluate(test, "test", best_model)

print("\nSample test predictions (unseen intents):")
for (src, ref), h in random.Random(0).sample(list(zip(test, hyp)), 12):
    print(f"  {'OK ' if ref == h else '   '}{src!r:40} want {ref!r:28} got {h!r}")

lookup = load_lookup()
print("\nDemo (dataset lookup / NAME / AGE first, then model):")
for s in ["My name is Ben", "I am 21 years old", "are you free this evening",
          "where is the hospital", "i want to drink tea", "please call my mother"]:
    g = to_gloss(s, lookup) or generate([normalize(s)], best_model)[0]
    print(f"  {s!r:30} -> {g}")

# skip interpreter teardown, which crashes in CUDA cleanup on Python 3.15 / Windows
sys.stdout.flush()
os._exit(0)
