from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from epicf_benchmark.data import FluDataset, all_indices


def collect_xy(dataset: FluDataset, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def collect_x(dataset: FluDataset, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs = []
    for x, _ in loader:
        xs.append(x.detach().cpu().numpy())
    return np.concatenate(xs, axis=0)


def flatten_features(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def kde_predict_mean(
    z_query: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    bandwidth: float,
    eps: float = 1e-12,
) -> np.ndarray:
    q2 = np.sum(z_query**2, axis=1, keepdims=True)
    t2 = np.sum(z_train**2, axis=1, keepdims=True).T
    dist2 = np.maximum(q2 + t2 - 2.0 * (z_query @ z_train.T), 0.0)
    log_kernel = -dist2 / (2.0 * bandwidth**2)
    log_kernel -= np.max(log_kernel, axis=1, keepdims=True)
    weights = np.exp(log_kernel)
    weights /= np.sum(weights, axis=1, keepdims=True) + eps
    y_flat = y_train.reshape(y_train.shape[0], -1)
    pred_flat = weights @ y_flat
    return pred_flat.reshape(z_query.shape[0], *y_train.shape[1:])


def train_kde(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _dataset(data_dir, args, "train")
    x_train, y_train = collect_xy(dataset, args.batch_size)
    z_train = flatten_features(x_train).astype(np.float64)

    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std = z_train.std(axis=0, keepdims=True) + 1e-8
    z_train_scaled = (z_train - z_mean) / z_std

    best_bandwidth, best_mse = None, float("inf")
    for bandwidth in args.bandwidth_grid:
        pred = kde_predict_mean(z_train_scaled, z_train_scaled, y_train, bandwidth)
        mse = np.mean((pred - y_train) ** 2)
        print(f"bandwidth={bandwidth:.3f} train_mse={mse:.6f}")
        if mse < best_mse:
            best_mse = mse
            best_bandwidth = bandwidth

    np.save(output_dir / "z_train.npy", z_train_scaled.astype(np.float32))
    np.save(output_dir / "y_train.npy", y_train.astype(np.float32))
    np.save(output_dir / "z_mean.npy", z_mean.astype(np.float32))
    np.save(output_dir / "z_std.npy", z_std.astype(np.float32))
    (output_dir / "best_bandwidth.txt").write_text(str(best_bandwidth), encoding="utf-8")
    print(f"Saved KDE artifacts to: {output_dir}")
    print(f"Best bandwidth: {best_bandwidth}")


def infer_kde(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _dataset(data_dir, args, args.split)
    x_query = collect_x(dataset, args.batch_size)
    z_query = flatten_features(x_query).astype(np.float64)

    z_train = np.load(model_dir / "z_train.npy").astype(np.float64)
    y_train = np.load(model_dir / "y_train.npy").astype(np.float64)
    z_mean = np.load(model_dir / "z_mean.npy").astype(np.float64)
    z_std = np.load(model_dir / "z_std.npy").astype(np.float64)
    bandwidth = float((model_dir / "best_bandwidth.txt").read_text(encoding="utf-8"))

    z_query_scaled = (z_query - z_mean) / z_std
    y_pred = kde_predict_mean(z_query_scaled, z_train, y_train, bandwidth)

    npy_path = output_dir / f"{args.prefix}.npy"
    csv_path = output_dir / f"{args.prefix}.csv"
    np.save(npy_path, y_pred)
    pd.DataFrame(y_pred.reshape(-1, 1)).to_csv(csv_path, header=False, index=False)
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")


def _dataset(data_dir: Path, args: argparse.Namespace, split: str) -> FluDataset:
    return FluDataset(
        data_dir / f"y_{split}.csv",
        data_dir / f"a_{split}.csv",
        data_dir / f"x_{split}.csv",
        all_indices(args.num_units),
        policy_dim=args.policy_dim,
        num_units=args.num_units,
        sequence_length=args.sequence_length,
        num_covariates=args.num_covariates,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--policy-dim", type=int, default=1)
    parser.add_argument("--num-units", type=int, default=158)
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument("--num-covariates", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the KDE baseline.")
    add_common_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bandwidth-grid",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    )
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KDE inference.")
    add_common_args(parser)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--prefix", default="kde_cf")
    return parser
