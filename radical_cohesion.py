"""Rebuild the per-radical cohesion tables in Appendix B from the character list.

Embeds all 6,306 characters with mBERT the same way the paper does — each character
on its own, final hidden layer, mean-pooled over [CLS], the character and [SEP] —
then averages the cosine similarity of every within-radical pair for each radical.

Run:  python radical_cohesion.py
Downloads mBERT on first use; a few minutes on CPU.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "bert-base-multilingual-cased"
BATCH = 256

# Kangxi radical number -> the form and gloss printed in the paper.
RADICALS = {
    15: ("冫", "ice"), 18: ("刂", "knife"), 19: ("力", "power"), 27: ("厂", "cliff"),
    30: ("口", "mouth"), 39: ("子", "child"), 44: ("尸", "corpse"), 46: ("山", "mountain"),
    53: ("广", "shelter"), 57: ("弓", "bow"), 60: ("彳", "step"), 62: ("戈", "halberd"),
    64: ("手", "hand"), 66: ("攴", "rap/strike"), 72: ("日", "sun"), 75: ("木", "tree"),
    86: ("火", "fire"), 93: ("牛", "ox"), 94: ("犭", "dog"), 96: ("玉", "jade"),
    104: ("疒", "sickness"), 108: ("皿", "dish"), 112: ("石", "stone"), 113: ("示", "spirit"),
    115: ("禾", "grain"), 116: ("穴", "cave"), 118: ("竹", "bamboo"), 119: ("米", "rice"),
    120: ("糸", "silk"), 124: ("羽", "feather"), 128: ("耳", "ear"), 142: ("虫", "insect"),
    145: ("衤", "clothing"), 147: ("见", "see"), 149: ("言", "speech"), 157: ("足", "foot"),
    159: ("车", "cart"), 162: ("辶", "walk"), 164: ("酉", "wine"), 167: ("金", "metal"),
    173: ("雨", "rain"), 184: ("食", "food"), 187: ("马", "horse"), 195: ("鱼", "fish"),
    196: ("鸟", "bird"),
}


def embed(chars):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    out = []
    for start in range(0, len(chars), BATCH):
        enc = tok(chars[start:start + BATCH], return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        print(f"  embedded {min(start + BATCH, len(chars))}/{len(chars)}", flush=True)
    vecs = np.concatenate(out).astype(np.float64)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main():
    ds = pd.read_csv("data/radical_dataset.csv").reset_index(drop=True)
    print(f"{len(ds)} characters, {ds['radical_number'].nunique()} radicals")

    vecs = embed(ds["char"].astype(str).tolist())

    rows = []
    for number, group in ds.groupby("radical_number"):
        idx = group.index.to_numpy()
        sims = vecs[idx] @ vecs[idx].T
        upper = np.triu_indices(len(idx), k=1)
        # RADICALS covers the 30 radicals tabulated in the paper. For the rest,
        # fall back to the Kangxi Radicals block character stored in the dataset.
        form, gloss = RADICALS.get(int(number),
                                   (str(group["kangxi_radical"].iloc[0]), ""))
        rows.append({"radical": form, "meaning": gloss, "number": int(number),
                     "size": len(idx), "pairs": len(upper[0]),
                     "cohesion": sims[upper].mean()})

    table = (pd.DataFrame(rows)
             .sort_values("cohesion", ascending=False)
             .reset_index(drop=True))
    table.index += 1

    cols = ["radical", "meaning", "size", "cohesion"]
    print("\nTop 20 radicals by cohesion (Table 6)")
    print(table.head(20)[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nBottom 10 radicals by cohesion (Table 7)")
    print(table.tail(10)[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nmean cohesion across the 68 radicals: {table['cohesion'].mean():.4f}")

    table.to_csv("radical_cohesion_mbert.csv", index_label="rank")
    print("wrote radical_cohesion_mbert.csv")


if __name__ == "__main__":
    main()
