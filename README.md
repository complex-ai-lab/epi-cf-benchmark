# [KDD 2026] Benchmarking Counterfactual Prediction in Epidemic Time Series with Time-Varying Interventions

This repository contains the machine-learning baselines and evaluation utilities for **EpiCF-Bench**, the benchmark introduced in:

> Wenhao Mu, Facundo Yan, Anik Mumssen, Marisa Eisenberg, and Alexander Rodriguez. 2026. Benchmarking Counterfactual Prediction in Epidemic Time Series with Time-Varying Interventions. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

EpiCF-Bench evaluates counterfactual prediction for epidemic time series under time-varying interventions. The benchmark is generated from a calibrated differentiable agent-based model (ABM), producing factual and counterfactual epidemic trajectories across 158 U.S. counties.

## Resources

- ABM simulator repository: [complex-ai-lab/epi-diff-abm](https://github.com/complex-ai-lab/epi-diff-abm)
- Benchmark dataset: [Zenodo record 20366265](https://zenodo.org/records/20366265)
- This repository: ML baselines, prediction scripts, and evaluation code

The ABM code and benchmark dataset are intentionally maintained outside this repository. This repo is meant to stay lightweight and focused on reproducing the learning-based benchmark experiments.

## Current Status

The first cleaned release includes:

- KDE baseline for single-policy and multi-policy settings
- Transformer baseline for single-policy and multi-policy settings
- RNN S-learner baseline for single-policy and multi-policy settings
- RNN T-learner baseline for single-policy and multi-policy settings
- Shared dataset loading code for ABM-generated CSV files
- Preprocessing and postprocessing scripts for benchmark data and model outputs
- Trajectory, distributional, predictive-interval, calibration, and CATE evaluation utilities

Additional baselines from the paper, including VAE, diffusion, and TECDE, will be added in later updates.

## Repository Layout

```text
.
├── experiments/
│   ├── single_policy/
│   │   ├── train_kde.py
│   │   ├── infer_kde.py
│   │   ├── train_transformer.py
│   │   ├── infer_transformer.py
│   │   └── train_st_learner.py
│   └── multi_policy/
│       ├── train_kde.py
│       ├── infer_kde.py
│       ├── train_transformer.py
│       ├── infer_transformer.py
│       └── train_st_learner.py
├── scripts/
│   ├── preprocess_data.py
│   ├── postprocess_predictions.py
│   ├── evaluate_predictions.py
│   ├── evaluate_distribution.py
│   └── evaluate_cate.py
└── src/
    └── epicf_benchmark/
        ├── data.py
        ├── evaluate.py
        ├── kde.py
        ├── models.py
        ├── postprocess.py
        ├── preprocessing.py
        ├── st_learner.py
        └── transformer.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer not to install the package, run scripts with:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Data

Download the benchmark data from Zenodo and place or symlink the processed CSV folders into the repository root:

```text
data_single_final/
├── y_train.csv
├── a_train.csv
├── x_train.csv
├── y_test.csv
├── a_test.csv
└── x_test.csv

data_multi_final/
├── y_train.csv
├── a_train.csv
├── x_train.csv
├── y_test.csv
├── a_test.csv
└── x_test.csv
```

The default experiments assume:

- sequence length: 168
- number of counties/units: 158
- covariate channels: 17
- single-policy treatment channels: 1
- multi-policy treatment channels: 2

## Preprocessing

The preprocessing scripts convert raw ABM output tables into the `y/a/x/xa` files consumed by the ML baselines.

For the single-policy setting:

```bash
python scripts/preprocess_data.py \
  --setting single \
  --input-csv data_single_final/all_county_data_extra_sp.csv \
  --output-dir data_single_final
```

For the multi-policy setting:

```bash
python scripts/preprocess_data.py \
  --setting multi \
  --input-csv data_multi_final/all_county_data_extra_mp.csv \
  --output-dir data_multi_final
```

Each run also writes `outcome_stats.json`, which stores the factual-outcome normalization parameters used by postprocessing.

## Running Baselines

### Single-Policy KDE

```bash
python experiments/single_policy/train_kde.py \
  --data-dir data_single_final \
  --output-dir kde_models_single

python experiments/single_policy/infer_kde.py \
  --data-dir data_single_final \
  --model-dir kde_models_single \
  --output-dir output_single \
  --split test \
  --prefix kde_cf
```

### Multi-Policy KDE

```bash
python experiments/multi_policy/train_kde.py \
  --data-dir data_multi_final \
  --output-dir kde_models_multi

python experiments/multi_policy/infer_kde.py \
  --data-dir data_multi_final \
  --model-dir kde_models_multi \
  --output-dir output_multi \
  --split test \
  --prefix kde_cf
```

### Single-Policy Transformer

```bash
python experiments/single_policy/train_transformer.py \
  --data-dir data_single_final \
  --output-dir models_single \
  --epochs 100

python experiments/single_policy/infer_transformer.py \
  --data-dir data_single_final \
  --model-path models_single/transformer.pt \
  --output-dir output_single \
  --split test \
  --prefix transformer_cf
```

### Multi-Policy Transformer

```bash
python experiments/multi_policy/train_transformer.py \
  --data-dir data_multi_final \
  --output-dir models_multi \
  --epochs 200

python experiments/multi_policy/infer_transformer.py \
  --data-dir data_multi_final \
  --model-path models_multi/transformer.pt \
  --output-dir output_multi \
  --split test \
  --prefix transformer_cf
```

### Single-Policy RNN S/T Learner

```bash
python experiments/single_policy/train_st_learner.py \
  --data-dir data_single_final \
  --model-dir models_single_final \
  --output-dir output_single_final \
  --epochs 200 \
  --seed 1 \
  --run-id 1_new
```

This writes files such as `slearner_rnn_mean_f_1_new.npy`, `slearner_rnn_mean_cf_1_new.npy`, `tlearner_rnn_mean_f_1_new.npy`, and `tlearner_rnn_mean_cf_1_new.npy`.

### Multi-Policy RNN S/T Learner

```bash
python experiments/multi_policy/train_st_learner.py \
  --data-dir data_multi_final \
  --model-dir models_multi_final \
  --output-dir output_multi_final \
  --epochs 200 \
  --seed 3 \
  --run-id 3_new
```

The multi-policy script trains on `y_train_3.csv`, `a_train_3.csv`, and `x_train_3.csv`, matching the original multi-policy S/T learner setup.

## Evaluation

## Postprocessing

The postprocessing script inverse-transforms normalized model outputs back to the original case-count scale, following the original `postprocess.ipynb` workflow.

```bash
python scripts/postprocess_predictions.py \
  --input output_single/transformer_cf.csv \
  --stats data_single_final/outcome_stats.json \
  --output-prefix output_single/transformer_cf_inverse \
  --num-units 158
```

For multi-policy factual predictions that need to be repeated across the three counterfactual policy alternatives, use `--tile 3`.

## Evaluation

For direct point-trajectory prediction metrics:

```bash
python scripts/evaluate_predictions.py \
  --y-true path/to/ground_truth.npy \
  --y-pred output_single/transformer_cf.npy \
  --output results/transformer_single_metrics.csv
```

For sampled predictions and distributional metrics:

```bash
python scripts/evaluate_distribution.py \
  --truth data_single_final/y_test_unnorm.csv \
  --pred output_single/cvae_cf_inverse.npy \
  --num-units 158 \
  --samples-per-unit 100 \
  --output results/cvae_single_distribution.csv
```

For CATE RMSE:

```bash
python scripts/evaluate_cate.py \
  --factual-truth data_single_final/y_train_unnorm.csv \
  --counterfactual-truth data_single_final/y_test_unnorm.csv \
  --factual-pred output_single/transformer_f_inverse.npy \
  --counterfactual-pred output_single/transformer_cf_inverse.npy \
  --num-units 158 \
  --output results/transformer_single_cate.csv
```

## Citation

```bibtex
@inproceedings{mu2026epicfbench,
  title = {Benchmarking Counterfactual Prediction in Epidemic Time Series with Time-Varying Interventions},
  author = {Mu, Wenhao and Yan, Facundo and Mumssen, Anik and Eisenberg, Marisa and Rodriguez, Alexander},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year = {2026}
}
```
