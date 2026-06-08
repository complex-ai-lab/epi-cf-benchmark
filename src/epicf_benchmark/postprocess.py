from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def inverse_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        raise ValueError("std must be nonzero for inverse z-score normalization.")
    return values * std + mean


def inverse_minmax(values: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    if y_max == y_min:
        raise ValueError("min and max must differ for inverse min-max normalization.")
    return values * (y_max - y_min) + y_min


def load_prediction(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    return pd.read_csv(path, header=None).values.reshape(-1)


def load_stats(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inverse_transform(values: np.ndarray, stats: dict) -> np.ndarray:
    transform = stats.get("transform", "zscore")
    if transform == "zscore":
        return inverse_zscore(values, float(stats["mean"]), float(stats["std"]))
    if transform == "minmax":
        return inverse_minmax(values, float(stats["min"]), float(stats["max"]))
    raise ValueError(f"Unknown transform: {transform}")


def save_prediction(
    values: np.ndarray,
    output_prefix: Union[str, Path],
    *,
    num_units: int,
    sequence_length: int,
) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    flat = values.reshape(-1)
    pd.DataFrame(flat).to_csv(output_prefix.with_suffix(".csv"), index=False, header=False)
    np.save(output_prefix.with_suffix(".npy"), flat.reshape(num_units, sequence_length))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse-transform normalized ML predictions.")
    parser.add_argument("--input", required=True, help="CSV or NPY prediction file.")
    parser.add_argument("--stats", required=True, help="outcome_stats.json from preprocessing.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--num-units", type=int, default=158)
    parser.add_argument("--sequence-length", type=int, default=168)
    parser.add_argument(
        "--tile",
        type=int,
        default=1,
        help="Optionally repeat rows before saving, useful for multi-policy factual predictions.",
    )
    args = parser.parse_args()

    values = inverse_transform(load_prediction(args.input), load_stats(args.stats))
    values = values.reshape(-1, args.sequence_length)
    if args.tile > 1:
        values = np.tile(values, (args.tile, 1))
    save_prediction(
        values,
        args.output_prefix,
        num_units=args.num_units * args.tile,
        sequence_length=args.sequence_length,
    )
    print(f"Saved: {Path(args.output_prefix).with_suffix('.csv')}")
    print(f"Saved: {Path(args.output_prefix).with_suffix('.npy')}")


if __name__ == "__main__":
    main()
