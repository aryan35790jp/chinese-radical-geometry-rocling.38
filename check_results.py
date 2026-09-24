"""Recompute every statistic in the paper from the stored arrays and check it.

Needs no model download. Run:  python check_results.py
"""

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"
MODELS = (("mBERT", "mbert"), ("Chinese-BERT", "chinese_bert"))
ok = fail = 0


def report(label, got, expected, tol):
    global ok, fail
    good = abs(got - expected) <= tol
    ok, fail = ok + good, fail + (not good)
    flag = "ok " if good else "OFF"
    print(f"  [{flag}] {label:42s} computed {got:<12.5g} paper {expected:<12.5g}")


def cohens_d(a, b):
    """Equal-weight variance average, as used in the paper."""
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def load(name):
    return np.load(f"{RESULTS}/{name}.npy").astype(np.float64)


main = pd.read_csv(f"{RESULTS}/main_results.csv").set_index("Model")
sem = pd.read_csv(f"{RESULTS}/semantic_control_results.csv")

print("Table 2 - corpus-scale radical cohesion")
for label, tag in MODELS:
    for metric, suffix, col in (("cosine", "", "Cos"), ("euclidean", "_euclid", "Euc")):
        intra = load(f"{tag}{suffix}_intra_pairs")
        inter = load(f"{tag}{suffix}_inter_pairs")
        row = main.loc[label]
        assert len(intra) == len(inter) == 3400, "expected 3,400-pair pools"
        report(f"{label} {metric} intra mean", intra.mean(), row[f"{col}_Intra"], 5e-5)
        report(f"{label} {metric} inter mean", inter.mean(), row[f"{col}_Inter"], 5e-5)
        report(f"{label} {metric} delta", intra.mean() - inter.mean(),
               row[f"{col}_Diff"], 5e-5)
        report(f"{label} {metric} Cohen's d", cohens_d(intra, inter), row[f"{col}_d"], 5e-4)
        _, p = stats.ttest_ind(intra, inter, equal_var=False)
        report(f"{label} {metric} Welch p", p, row[f"{col}_p"], row[f"{col}_p"] * 1e-3)

print("\nTable 9 - bootstrap confidence intervals")
for label, tag in MODELS:
    for metric, suffix, col in (("cosine", "", "Cos"), ("euclidean", "_euclid", "Euc")):
        boot = load(f"{tag}{suffix}_bootstrap")
        assert len(boot) == 1000, "expected 1,000 resamples"
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row = main.loc[label]
        report(f"{label} {metric} CI low", lo, row[f"{col}_CI_lo"], 5e-5)
        report(f"{label} {metric} CI high", hi, row[f"{col}_CI_hi"], 5e-5)

print("\nTable 8 - permutation tests")
for label, tag in MODELS:
    for metric, suffix, col in (("cosine", "", "Cos"), ("euclidean", "_euclid", "Euc")):
        perm = load(f"{tag}{suffix}_permutation_scores")
        intra = load(f"{tag}{suffix}_intra_pairs")
        inter = load(f"{tag}{suffix}_inter_pairs")
        assert len(perm) == 1000, "expected 1,000 shuffles"
        obs = intra.mean() - inter.mean()
        p = (np.sum(perm >= obs) + 1) / (len(perm) + 1)
        key = "Perm_p_cos" if metric == "cosine" else "Perm_p_euc"
        report(f"{label} {metric} permutation p", p, main.loc[label, key], 5e-5)
        print(f"        null SD {perm.std(ddof=1):.4f}")

print("\nTable 10 - Holm-Bonferroni correction")
raw = sorted((main.loc[model, f"{col}_p"], model, metric)
             for model, _tag in MODELS
             for col, metric in (("Cos", "cosine"), ("Euc", "euclidean")))
for i, (p, model, metric) in enumerate(raw):
    thresh = 0.05 / (len(raw) - i)
    verdict = "passes" if p < thresh else "FAILS"
    print(f"  [{'ok ' if p < thresh else 'OFF'}] {model} {metric:10s} "
          f"raw p {p:.3g} < {thresh:.4f}  {verdict}")
    ok, fail = ok + (p < thresh), fail + (p >= thresh)

print("\nTable 3 - confound checks")
for label, tag in MODELS:
    coh = load(f"{tag}_rad_cohesions")
    sizes = load(f"{tag}_rad_sizes")
    assert len(coh) == len(sizes) == 68, "expected 68 radicals"
    rho, p = stats.spearmanr(coh, sizes)
    report(f"{label} cohesion vs group size rho", rho, main.loc[label, "Size_rho"], 5e-4)
    report(f"{label} cohesion vs group size p", p, main.loc[label, "Size_p"], 5e-4)
    intra = load(f"{tag}_intra_pairs")
    same = np.allclose(intra.reshape(68, 50).mean(axis=1), coh, atol=1e-5)
    print(f"  [{'ok ' if same else 'OFF'}] {label} per-radical cohesion equals the "
          f"mean of that radical's 50 sampled pairs")
    ok, fail = ok + same, fail + (not same)

print("\nTable 4 - semantic control")
FIELDS = ("animals", "water", "wood", "metal")
for label, tag in MODELS:
    intra = load(f"{tag}_semantic_control_intra")
    cross = load(f"{tag}_semantic_control_cross")
    perm = load(f"{tag}_semantic_control_perm")
    assert len(intra) == 40 and len(cross) == 100, "expected 40 intra / 100 cross"
    assert len(perm) == 5000, "expected 5,000 shuffles"
    for i, field in enumerate(FIELDS):
        bi, bc = intra[i * 10:(i + 1) * 10], cross[i * 25:(i + 1) * 25]
        row = sem[(sem.Model == label) & (sem.Field.str.startswith(field))].iloc[0]
        report(f"{label} {field} intra", bi.mean(), row.Intra, 5e-5)
        report(f"{label} {field} cross", bc.mean(), row.Cross, 5e-5)
        report(f"{label} {field} d", cohens_d(bi, bc), row.d, 5e-4)
    pooled = sem[(sem.Model == label) & (sem.Field == "POOLED")].iloc[0]
    report(f"{label} pooled intra", intra.mean(), pooled.Intra, 5e-5)
    report(f"{label} pooled cross", cross.mean(), pooled.Cross, 5e-5)
    report(f"{label} pooled d", cohens_d(intra, cross), pooled.d, 5e-4)
    _, p = stats.ttest_ind(intra, cross, equal_var=False)
    report(f"{label} pooled Welch p", p, pooled.p, 5e-4)
    obs = intra.mean() - cross.mean()
    print(f"        permutation p {(np.sum(perm >= obs) + 1) / (len(perm) + 1):.4f}, "
          f"null SD {perm.std(ddof=1):.4f}")

print(f"\n{ok} checks matched, {fail} did not.")
