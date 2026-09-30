# TRACE — Temporally Registered, Acquisition-Consistent change Encoding

Reference implementation for the manuscript *"Correspondence-Gated Longitudinal Change Tokens with
Acquisition-Counterfactual Consistency for Vendor-Robust Mammographic Cancer Detection and Conformal
Interval-Risk Referral"*. Everything needed to regenerate the in-silico cohort, pretrain the frozen
encoder, train TRACE and every baseline/ablation, and rebuild every table and figure is in this folder.
The folder is flat: all scripts import each other from the same directory.

## Requirements
Python 3.10+, PyTorch >= 2.1, NumPy, SciPy, scikit-learn, Matplotlib, python-docx, lxml, pandoc
(only for rebuilding the manuscript). `pip install -r requirements.txt`.
All experiments in the paper were run on a 2-core CPU; no GPU is required.

## Pipeline
```bash
mkdir -p ../data ../res ../fig
# 1. simulate cohorts (development V1-V3, external V4/V5, disjoint pretraining cohort)
python generate_cohorts.py ../data pre dev ext
python generate_cohorts.py ../data devu            # test priors re-rendered through unseen chains V4/V5
python generate_cohorts.py ../data devw 2.0 3.0    # test priors with 2x / 3x positioning deformation
# 2. pretrain the encoder on the disjoint pretraining cohort, then freeze it and cache features
python pretrain_encoder.py
python cache_features.py dev_img dev_cf ext_img devu_cf devw20_cf devw30_cf
# 3. classical baseline and all learned configurations (name, seed, epochs)
python classical_baseline.py
python run_queue.py jobs.txt 2          # or: python train.py "Proposed" 0 6
# 4. aggregate metrics, statistics, conformal layer
python evaluate_all.py
python bench.py
# 5. figures
python fig_architecture.py && python fig_cohort.py && python fig_qual.py && python fig_results.py
```

## Files
| file | purpose |
|---|---|
| `simulator.py` | power-law breast anatomy, lesion histories, positioning warps, five acquisition chains |
| `generate_cohorts.py` | development / external / pretraining cohorts, acquisition counterfactuals |
| `common.py` | patient-level splits, encoder, acquisition-style augmentation |
| `pretrain_encoder.py` | weakly supervised + contrastive encoder pretraining (disjoint cohort) |
| `cache_features.py` | frozen-encoder feature grids |
| `models.py` | TRACE (correspondence, confidence, gated change, tokens, relational transformer) and baselines |
| `train.py` | objectives (supervised, correspondence, saliency, ACC, contrastive invariance), training loop |
| `classical_baseline.py` | hand-crafted temporal CAD features + gradient boosting |
| `metrics.py` | AUC/AP/sensitivity, ECE, localisation, Mondrian conformal sets, typicality gate, CRC, DeLong, Holm |
| `evaluate_all.py` | result tables, strata, significance, conformal summaries |
| `bench.py` | parameters, FLOPs, CPU latency |
| `fig_*.py`, `diagram.py` | figures (grayscale) |
| `jobs.txt`, `run_queue.py` | exact list of runs reported in the paper |

## Notes
* Splits are at the woman level (`common.splits`), so no woman contributes to more than one split.
* External vendors V4/V5 are never used before external testing; counterfactual renderings use V1–V3 only.
* The cohort is simulated; results do not transfer to clinical performance without clinical validation.

## Evaluation sets
`test` (in-distribution), `testsw` (priors re-rendered through another development chain), `testswu`
(priors re-rendered through unseen chains), `testw20`/`testw30` (prior misregistration x2/x3), `ext`
(external cohort on unseen chains). All are produced by `train.py` for every run.
