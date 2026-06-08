from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wasserstein(array1: np.ndarray, array2: np.ndarray) -> float:
    _assert_same_shape(array1, array2)
    return float(wasserstein_distance(array1.ravel(), array2.ravel()))


def empirical_wasserstein(array1: np.ndarray, array2: np.ndarray) -> float:
    _assert_same_shape(array1, array2)
    return float(np.mean(np.abs(array1 - array2)))


def wasserstein_by_feature(array1: np.ndarray, array2: np.ndarray) -> float:
    _assert_same_shape(array1, array2)
    return float(
        np.mean(
            [wasserstein_distance(array1[:, dim], array2[:, dim]) for dim in range(array1.shape[1])]
        )
    )


def calculate_rmse(array1: np.ndarray, array2: np.ndarray, *, mean_over_samples: bool = False) -> float:
    _assert_same_shape(array1, array2)
    if mean_over_samples:
        array1 = np.mean(array1, axis=0)
        array2 = np.mean(array2, axis=0)
    return float(np.sqrt(np.mean((array1 - array2) ** 2)))


def predictive_interval_coverage(
    truth: np.ndarray,
    samples: np.ndarray,
    alphas: tuple[float, ...] = (0.25, 0.10, 0.05),
) -> dict[str, float]:
    coverages = {}
    for alpha in alphas:
        lower = np.quantile(samples, q=alpha / 2, axis=0)
        upper = np.quantile(samples, q=1 - alpha / 2, axis=0)
        coverages[f"PI{int((1 - alpha) * 100)}"] = float(
            np.mean((truth >= lower) & (truth <= upper))
        )
    return coverages


def correlation(array1: np.ndarray, array2: np.ndarray) -> float:
    _assert_same_shape(array1, array2)
    values = []
    for x, y in zip(array1, array2):
        if np.std(x) == 0 or np.std(y) == 0:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values))


def calibration_score(
    truth: np.ndarray,
    samples: np.ndarray,
    alphas: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02),
) -> float:
    observed = []
    for alpha in alphas:
        lower = np.quantile(samples, q=alpha / 2, axis=0)
        upper = np.quantile(samples, q=1 - alpha / 2, axis=0)
        observed.append(np.mean((truth >= lower) & (truth <= upper)))
    expected = 1 - np.array(alphas)
    return float(np.mean(np.abs(np.array(observed) - expected)))


def calculate_cate(array_treated: np.ndarray, array_control: np.ndarray) -> np.ndarray:
    _assert_same_shape(array_treated, array_control)
    return array_treated - array_control


def evaluate_distribution(
    truth: np.ndarray,
    prediction_samples: np.ndarray,
    *,
    num_units: int,
    samples_per_unit: int = 100,
) -> dict[str, float]:
    truth = truth.reshape(num_units, -1)
    prediction_samples = prediction_samples.reshape(-1, truth.shape[1])
    per_metric: dict[str, list[float]] = {
        "wasserstein": [],
        "empirical_wasserstein": [],
        "t_wasserstein": [],
        "rmse": [],
        "PI75": [],
        "PI90": [],
        "PI95": [],
        "correlation": [],
        "calibration_score": [],
    }

    for unit_idx in range(num_units):
        pred_slice = prediction_samples[unit_idx::num_units, :]
        if len(pred_slice) != samples_per_unit:
            samples_per_unit = len(pred_slice)
        truth_slice = np.repeat(truth[unit_idx].reshape(1, -1), len(pred_slice), axis=0)
        scale = max(float(np.max(truth_slice) - np.min(truth_slice)), 1.0)
        per_metric["wasserstein"].append(wasserstein(truth_slice, pred_slice))
        per_metric["empirical_wasserstein"].append(empirical_wasserstein(truth_slice, pred_slice))
        per_metric["t_wasserstein"].append(wasserstein_by_feature(truth_slice, pred_slice) / scale)
        per_metric["rmse"].append(calculate_rmse(truth_slice, pred_slice) / scale)
        pi = predictive_interval_coverage(truth_slice, pred_slice)
        per_metric["PI75"].append(pi["PI75"])
        per_metric["PI90"].append(pi["PI90"])
        per_metric["PI95"].append(pi["PI95"])
        per_metric["correlation"].append(correlation(truth_slice, pred_slice))
        per_metric["calibration_score"].append(calibration_score(truth_slice, pred_slice))

    return {name: float(np.mean(values)) for name, values in per_metric.items()}


def evaluate_cate_rmse(
    factual_truth: np.ndarray,
    counterfactual_truth: np.ndarray,
    factual_pred: np.ndarray,
    counterfactual_pred: np.ndarray,
    *,
    num_units: int,
) -> float:
    factual_truth = factual_truth.reshape(num_units, -1)
    counterfactual_truth = counterfactual_truth.reshape(num_units, -1)
    factual_pred = factual_pred.reshape(-1, factual_truth.shape[1])
    counterfactual_pred = counterfactual_pred.reshape(-1, factual_truth.shape[1])

    scores = []
    for unit_idx in range(num_units):
        factual_pred_slice = factual_pred[unit_idx::num_units, :]
        counterfactual_pred_slice = counterfactual_pred[unit_idx::num_units, :]
        repeats = len(factual_pred_slice)
        factual_truth_slice = np.repeat(factual_truth[unit_idx].reshape(1, -1), repeats, axis=0)
        counterfactual_truth_slice = np.repeat(
            counterfactual_truth[unit_idx].reshape(1, -1), repeats, axis=0
        )
        cate_truth = calculate_cate(factual_truth_slice, counterfactual_truth_slice)
        cate_pred = calculate_cate(factual_pred_slice, counterfactual_pred_slice)
        scale = max(float(np.max(cate_truth) - np.min(cate_truth)), 1.0)
        scores.append(calculate_rmse(cate_truth, cate_pred) / scale)
    return float(np.mean(scores))


def policy_regret(
    y_factual_pred: np.ndarray,
    y_counterfactual_pred: np.ndarray,
    y_factual_true: np.ndarray,
    y_counterfactual_true: np.ndarray,
    factual_policy: np.ndarray,
    *,
    week_idx: int = 0,
    intervention_cost: float = 3232.32,
) -> pd.DataFrame:
    rows = []
    for idx in range(len(y_factual_pred)):
        if factual_policy[idx, week_idx] == 0:
            pred_y0, pred_y1 = y_factual_pred[idx], y_counterfactual_pred[idx]
            true_y0, true_y1 = y_factual_true[idx], y_counterfactual_true[idx]
        else:
            pred_y0, pred_y1 = y_counterfactual_pred[idx], y_factual_pred[idx]
            true_y0, true_y1 = y_counterfactual_true[idx], y_factual_true[idx]

        pred_loss_0 = np.sum(pred_y0[week_idx:])
        pred_loss_1 = np.sum(pred_y1[week_idx:]) + intervention_cost
        selected_policy = int(pred_loss_1 < pred_loss_0)

        true_loss_0 = np.sum(true_y0[week_idx:])
        true_loss_1 = np.sum(true_y1[week_idx:]) + intervention_cost
        optimal_policy = int(true_loss_1 < true_loss_0)

        selected_true_loss = true_loss_1 if selected_policy else true_loss_0
        optimal_true_loss = min(true_loss_0, true_loss_1)
        rows.append(
            {
                "unit_id": idx,
                "selected_policy": selected_policy,
                "optimal_policy": optimal_policy,
                "selected_true_loss": selected_true_loss,
                "optimal_true_loss": optimal_true_loss,
                "regret": selected_true_loss - optimal_true_loss,
                "is_optimal": selected_policy == optimal_policy,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate point trajectory prediction files.")
    parser.add_argument("--y-true", required=True)
    parser.add_argument("--y-pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    y_true = _load_array(args.y_true)
    y_pred = _load_array(args.y_pred)
    metrics = pd.DataFrame([{"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved: {output}")


def _load_array(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    return pd.read_csv(path, header=None).values.reshape(-1)


def _assert_same_shape(array1: np.ndarray, array2: np.ndarray) -> None:
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: {array1.shape} != {array2.shape}")


if __name__ == "__main__":
    main()
