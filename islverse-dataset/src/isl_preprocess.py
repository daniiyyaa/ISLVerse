"""Sentence -> ISL gloss with NAME and AGE slot filling.

Run from the repo root:
    python src/isl_preprocess.py

Use in code:
    from isl_preprocess import load_lookup, to_gloss
    lookup = load_lookup("data/isl_dataset_full.csv")
    to_gloss("My name is Ben", lookup)      # -> "MY NAME B-E-N"
    to_gloss("I am 21 years old", lookup)   # -> "I 21"
"""
import re
import pandas as pd

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}

NAME_RE = re.compile(
    r"^(?:my name is|i am called|they call me|you can call me|you may call me|"
    r"people call me|everyone calls me) ([a-z]+(?: [a-z]+)?)$")
AGE_RES = [re.compile(p) for p in (
    r"^i am (\d{1,3})$",
    r"^my age is (\d{1,3})(?: years)?$",
    r"^i am (\d{1,3}) years(?: of age| old)?(?: now)?$",
    r"^i have completed (\d{1,3}) years$")]


def normalize(sentence):
    s = re.sub(r"[^\w\s()-]", "", sentence.lower()).strip()
    return re.sub(r"\s+", " ", s)


def numbers_to_digits(s):
    """'i am twenty one years old' -> 'i am 21 years old'"""
    out, toks, i = [], s.split(), 0
    while i < len(toks):
        t = toks[i]
        if t in TENS:
            val = TENS[t]
            if i + 1 < len(toks) and toks[i + 1] in UNITS and 1 <= UNITS[toks[i + 1]] <= 9:
                val += UNITS[toks[i + 1]]; i += 1
            out.append(str(val))
        elif t in UNITS:
            out.append(str(UNITS[t]))
        else:
            out.append(t)
        i += 1
    return " ".join(out)


def load_lookup(path="data/isl_dataset_full.csv"):
    df = pd.read_csv(path, keep_default_na=False)
    lookup = {}
    for sent, gloss in zip(df["English Sentence"], df["ISL Gloss"]):
        lookup.setdefault(str(sent).strip().lower(), gloss)   # first gloss wins
    return lookup


def fingerspell(name):
    return " ".join("-".join(w.upper()) for w in name.split())   # ben -> B-E-N


def to_gloss(sentence, lookup):
    s = normalize(sentence)
    if s in lookup:                              # 1) exact dataset match wins
        return lookup[s]
    m = NAME_RE.match(s)                         # 2) NAME slot
    if m:
        return lookup["my name is xxxxxxxx"].replace("XXXXXXXX", fingerspell(m.group(1)))
    s_num = numbers_to_digits(s)                 # 3) AGE slot (digits or number words)
    for rx in AGE_RES:
        m = rx.match(s_num)
        if m:
            return lookup["i am (age)"].replace("(AGE)", m.group(1))
    return None                                  # 4) unknown -> model / fallback


if __name__ == "__main__":
    lk = load_lookup()
    for t in ["My name is Ben", "they call me Ben Carter", "I am 21 years old",
              "i am twenty one", "I am hungry", "hello", "something unseen"]:
        print(f"{t!r:28} -> {to_gloss(t, lk)}")
