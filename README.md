# Continual Adversarial Training for Attack-Incremental IoT Intrusion Detection

Code, audit tooling and per-run results for the paper. Every number in
the manuscript is regenerated from the CSVs in `results/` by the scripts
in `analysis/`, so the paper and the data cannot drift apart.

## The audit tool is the part most likely to be useful to you

`audit/audit_leakage.py` measures two things about any tabular IDS
dataset, and neither is routinely reported in this literature:

- **Record diversity.** How many *distinct* feature vectors each class
  actually contains.
- **Held-out overlap.** What share of held-out rows are duplicates of
  training rows, under the split you actually use.

Running it on our own pipeline is what produced the corrections behind
this version of the paper:

```bash
python audit/audit_leakage.py --dirs path/to/balanced_subsets
```

On the four benchmarks we used it reports:

| dataset | distinct records | held-out overlap |
|---|---|---|
| CICIoT2023 | 98.1% | 2.9% |
| CICIoMT2024 | 93.6% | 8.9% |
| TON_IoT | 90.8% | 12.5% |
| **CICIoV2024** | **0.3%** | **99.9%** |

CICIoV2024 in the flat CAN-frame representation holds 2 distinct vectors
for Spoofing-GAS, 3 for Spoofing-STEERING-WHEEL and 21 for DoS. No
train/test split of it can be disjoint, which is why the paper excludes
it. That is a property of the representation, not of any one study, so
it applies to other work using it as well.

## What is here

```
cat/         the method and the training/evaluation pipeline
audit/       evaluation-integrity tooling (leakage, diversity, cost)
analysis/    statistics, tables and figures, all generated from results/
scripts/     data preparation and experiment runners
results/     320 runs: one master CSV per experimental stage
```

`results/` is the full record: one row per (dataset, seed, backbone,
variant) with clean, robust and continual-learning metrics. The stages
map to the paper as follows.

| file | what it contains |
|---|---|
| `s1_primary.csv` | CAT vs sequential AT, fixed backbone (H1) |
| `s3_ablation.csv` | EWC / distillation / replay, one at a time |
| `s6_method.csv` | the proposed method and its adversary-aware variant |
| `s7_arch_cat.csv` | the method on six backbones (architecture claim) |
| `s4_noat.csv`, `s8_seq_noat.csv` | the two beta=0 controls |
| `s5_sota_*.csv` | BiC and robust bias correction |
| `complexity_a100.csv` | parameters, PGD multiplier, memory, latency |

## Reproducing the paper's numbers

```bash
pip install -r requirements.txt

python analysis/make_tables_r2.py        # the LaTeX tables, from results/
python analysis/make_figures_r2.py       # the result figures
python analysis/analyze_r2.py --runs "results/s*.csv"   # the statistics
python analysis/check_consistency.py     # cross-checks the manuscript
```

`check_consistency.py` recomputes every headline quantity from the CSVs
and scans the manuscript for numbers that no longer match. It exists
because the previous version of this paper quoted a p-value that could
not be traced to any test; checking by hand would have fixed the
instance and left the mechanism in place.

## Reproducing the experiments

The subsets are built from the published datasets rather than shipped,
so the split-before-balance order is reproducible rather than taken on
trust:

```bash
python scripts/make_balanced_subset.py --total 100000 --no-oversample \
       --out balanced_data_ss100k
bash scripts/run_full_r2.sh
```

Runners are resumable: re-running after an interruption skips configs
already recorded as `ok` and retries ones that failed.

## Caveats worth knowing before you build on this

- Robustness numbers describe a **constraint-respecting feature-space**
  adversary: perturbations are projected onto per-feature bounds and
  integrality, and all satisfy them. This does not model dependence
  between features, and no perturbed vector is shown to correspond to a
  sendable packet.
- Claims rest on **two** benchmarks. CICIoV2024 is excluded as above;
  CICIoT2023 is reported per seed because its seed-to-seed variation is
  an order of magnitude larger than the other two.
- Three runs of the MLP backbone on the longest stream failed on an
  A100 with a device-side assertion and completed on a consumer GPU.
  This is a reduced-mantissa arithmetic sensitivity, not a property of
  the method, and those cells were re-run on the consumer GPU.
