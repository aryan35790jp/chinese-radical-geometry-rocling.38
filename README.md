# Radical-Aligned Structure in Multilingual Transformer Representations of Chinese Characters


- **11 models** spanning multilingual / Chinese / Japanese / glyph-aware /
  pure-vision baselines (`mBERT`, `Chinese-BERT`, `MacBERT`, `XLM-R base/large`,
  `ERNIE 3.0`, `UER-tiny/small`, `ChineseBERT-glyph`, `JP-BERT char/subword`,
  rendered-PNG → frozen ResNet-18).
- **All hidden layers** for every model, three pool types each (`char`,
  `mean`, `cls`).
- **Anisotropy correction** (Mu & Viswanath all-but-the-top) before every
  cosine measurement.
- **20+ semantic fields** for the controlled comparison, generated from a
  hand-curated mixed-radical taxonomy with a fallback path that doesn't
  require OpenHowNet.
- **Linear probes** for radical category and semantic field at every layer.
- **Phonetic vs semantic radical role** split via CHISE IDS.
- **Cross-script replication** on Japanese kanji.
- **Co-occurrence / PMI variance decomposition** that quantifies how much
  of the radical effect is form, semantics, distributional context, or
  frequency.
- **Mikolov-style orthographic arithmetic** and **geometric activation
  patching** as causal-flavored interventions.
- **Sentential-context analysis** comparing isolated vs in-sentence
  embeddings.
- **Downstream validation** against PKU-500 word similarity.


| Step                     | Compute            | RAM    | Runtime          |
|--------------------------|--------------------|--------|------------------|
| `extract_embeddings.py`  | GPU (A100 ideal)   | 16 GB  | 3 h A100 / 15 h T4 |
| `cooccurrence_baseline.py` (first run) | CPU, network | 8 GB   | 30 min for PMI build |
| `sentential_context.py`  | GPU                | 16 GB  | 90 min total     |
| `layer_wise_analysis.py` | CPU                | 8 GB   | 1 hr             |
| everything else          | CPU                | <8 GB  | minutes          |

## Statistical methods

- **Anisotropy correction**: mean-centering, per-coordinate standardization,
  removal of top-k principal components (Mu & Viswanath 2018,
  "All-But-the-Top"). Default k=2.
- **Cohen's d** with pooled unbiased standard deviation.
- **Welch's unequal-variance t-test** for the primary comparison.
- **Permutation test** (1,000 shuffles for corpus-scale, 5,000 for the
  semantic control). Continuity-corrected one-sided p.
- **Bootstrap 95% CI** (1,000 resamples) for the mean difference.
- **Holm–Bonferroni** correction across all primary comparisons.
- **Representational Similarity Analysis (RSA)**: Spearman correlation
  between the embedding RDM and the binary same-radical RDM.
- **Variance decomposition**: OLS of `bert_cosine ~ same_radical + ppmi +
  freq_diff + stroke_diff` on a 200k-pair sample, partial R² for each
  predictor.

## License

Research use. Dataset derived from Unicode Unihan.
