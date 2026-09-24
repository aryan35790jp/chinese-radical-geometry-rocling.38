# chinese-radical-geometry

Data and code for the paper *Radical-Aligned Structure in Multilingual Transformer
Representations of Chinese Characters: A Controlled Empirical Study* (ROCLING 2026).

## The short version

Chinese characters are filed under Kangxi radicals, and a radical often hints at what
a character means — 氵 shows up in 河 (river), 湖 (lake), 泪 (tears). We asked whether
mBERT and Chinese-BERT place characters that share a radical closer together in
embedding space.

Across 6,306 characters and 68 radicals, they do. The effect is reliable but small,
with Cohen's *d* between 0.06 and 0.14, and it holds for both models under both cosine
and Euclidean distance.

Then we ran the control that matters. Instead of comparing everything to everything,
we compared same-radical and different-radical characters that already belong to the
same meaning field — animals, water, wood, metal. The effect vanished (*d* ≈ −0.1,
*p* > 0.5). So what looks like radical structure is better explained by meaning:
radicals correlate with meaning, models organise by meaning, and the radical signal
comes along for the ride.

The negative control is the point of the paper, not a footnote.

## What's in here

```
data/
  radical_dataset.csv     6,306 characters with their Kangxi radical, stroke count,
                          group size and a frequency proxy. This is the filtered set
                          every number in the paper is computed on.
  radical_summary.csv     per-radical character counts (68 rows)

results/
  main_results.csv        Table 2 and Table 3: means, deltas, Cohen's d, p-values,
                          bootstrap intervals, confound correlations
  semantic_control_results.csv   Table 4: the four semantic fields plus pooled rows
  *_intra_pairs.npy       the 3,400 sampled within-radical similarities per model
  *_inter_pairs.npy       the 3,400 sampled between-radical similarities
  *_euclid_*.npy          the same two pools under Euclidean distance
  *_bootstrap.npy         1,000 bootstrap resamples of the intra-inter difference
  *_permutation_scores.npy   1,000 label shuffles
  *_rad_cohesions.npy     mean within-radical similarity for each of the 68 radicals
  *_rad_sizes.npy         how many characters each radical has
  *_semantic_control_*.npy   the 40 within-radical and 100 cross-radical pairs from
                          the control experiment, plus its 5,000 shuffles

figures/                  the six figures that appear in the paper

check_results.py          recomputes every statistic in the paper from the arrays
                          above and reports whether it matches
radical_cohesion.py       rebuilds the per-radical cohesion tables in Appendix B
                          from data/radical_dataset.csv and mBERT
```

## Reproducing the numbers

```bash
pip install -r requirements.txt

python check_results.py       # no model download, runs in seconds
python radical_cohesion.py    # downloads mBERT, a few minutes on CPU
```

`check_results.py` reads the stored arrays and re-derives the Welch tests, Cohen's *d*,
the bootstrap intervals, the permutation null distributions and the Holm correction,
then checks each against what the paper reports. `radical_cohesion.py` starts from the
character list instead and recomputes cohesion for all 68 radicals from scratch.

## A few notes on the data

The character set comes from the Unicode Unihan database. We don't redistribute
Unihan here — download it from <https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip>
if you want to rebuild the character list yourself. `data/radical_dataset.csv` is the
finished product.

Three filters produced it: the radical must have at least 20 characters in the set,
each character must be a single token under both the mBERT and Chinese-BERT
tokenizers, and variant codepoints for the same character are merged.

The corpus-scale tests use balanced samples of 3,400 within-radical pairs (50 per
radical) and 3,400 between-radical pairs rather than all 19.9 million available pairs.
With pools that large every difference is significant regardless of size, so the
effect sizes and bootstrap intervals carry the information, not the *p*-values.

The full 6,306 × 6,306 similarity matrices are about 160 MB each, so they aren't in
the repo. `radical_cohesion.py` rebuilds what it needs.

Embeddings come from the final hidden layer, mean-pooled over `[CLS]`, the character
token and `[SEP]`, with the character fed in on its own.

## Models

- `bert-base-multilingual-cased`
- `hfl/chinese-bert-wwm-ext`

## Citation

```bibtex
@inproceedings{maity2026radical,
  title     = {Radical-Aligned Structure in Multilingual Transformer Representations
               of Chinese Characters: A Controlled Empirical Study},
  author    = {Maity, Aryan},
  booktitle = {Proceedings of the 37th Conference on Computational Linguistics and
               Speech Processing (ROCLING)},
  year      = {2026}
}
```
