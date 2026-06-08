from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


AGE_COLUMNS = ["U19", "20t29", "30t39", "40t49", "50t64", "65A"]
MOBILITY_COLUMNS = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]
SINGLE_SERM_FACTUAL = [
    "susceptible_sch=0-occ=F",
    "exposed_sch=0-occ=F",
    "recovered_sch=0-occ=F",
    "dead_sch=0-occ=F",
]
SINGLE_SERM_COUNTERFACTUAL = [
    "susceptible_sch=0-occ=CF",
    "exposed_sch=0-occ=CF",
    "recovered_sch=0-occ=CF",
    "dead_sch=0-occ=CF",
]
MULTI_COUNTERFACTUALS = ["CF-F", "F-CF", "CF-CF"]
MULTI_SERM_FACTUAL = ["susceptible_F-F", "exposed_F-F", "recovered_F-F", "dead_F-F"]


@dataclass(frozen=True)
class OutcomeStats:
    transform: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


def minmax_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        raise ValueError(f"Min-max failed for column {series.name!r}.")
    return (values - vmin) / (vmax - vmin)


def zscore_outcomes(factual: pd.Series, *counterfactuals: pd.Series) -> tuple[list[pd.Series], OutcomeStats]:
    factual = factual.astype(float)
    mean = float(factual.mean())
    std = float(factual.std())
    if std == 0:
        raise ValueError("Factual outcome has std=0 and cannot be z-score normalized.")
    normalized = [(factual - mean) / std]
    normalized.extend((cf.astype(float) - mean) / std for cf in counterfactuals)
    return normalized, OutcomeStats(transform="zscore", mean=mean, std=std)


def build_base_covariates(df: pd.DataFrame) -> pd.DataFrame:
    population = minmax_series(df["population_size"]).rename("population_size")
    age = df[AGE_COLUMNS].astype(float) / 100.0
    mobility = df[MOBILITY_COLUMNS].apply(minmax_series)
    return pd.concat([population, age, mobility], axis=1)


def preprocess_single(input_csv: Union[str, Path], output_dir: Union[str, Path]) -> OutcomeStats:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    y_norm, stats = zscore_outcomes(df["0-F"], df["0-CF"])
    y_norm[0].to_csv(output_dir / "y_train.csv", index=False, header=False)
    y_norm[1].to_csv(output_dir / "y_test.csv", index=False, header=False)
    df["0-F"].astype(float).to_csv(output_dir / "y_train_unnorm.csv", index=False, header=False)
    df["0-CF"].astype(float).to_csv(output_dir / "y_test_unnorm.csv", index=False, header=False)

    a_train = df["treatment"].astype(int)
    a_test = 1 - a_train
    a_train.to_csv(output_dir / "a_train.csv", index=False, header=False)
    a_test.to_csv(output_dir / "a_test.csv", index=False, header=False)

    base = build_base_covariates(df)
    serm_f = df[SINGLE_SERM_FACTUAL].astype(float).copy()
    serm_cf = df[SINGLE_SERM_COUNTERFACTUAL].astype(float).copy()
    serm_f.columns = ["S", "E", "R", "M"]
    serm_cf.columns = ["S", "E", "R", "M"]
    x_train = pd.concat([base, serm_f], axis=1)
    x_test = pd.concat([base, serm_cf], axis=1)
    x_train.to_csv(output_dir / "x_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)

    write_xa_files(output_dir, policy_dim=1)
    write_stats(output_dir, stats)
    return stats


def preprocess_multi(input_csv: Union[str, Path], output_dir: Union[str, Path]) -> OutcomeStats:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    y_norm, stats = zscore_outcomes(
        df["F-F"],
        df["CF-F"],
        df["F-CF"],
        df["CF-CF"],
    )
    y_norm[0].to_csv(output_dir / "y_train.csv", index=False, header=False)
    pd.concat(y_norm[1:], axis=0, ignore_index=True).to_csv(
        output_dir / "y_test.csv", index=False, header=False
    )
    df["F-F"].astype(float).to_csv(output_dir / "y_train_unnorm.csv", index=False, header=False)
    pd.concat(
        [df["CF-F"].astype(float), df["F-CF"].astype(float), df["CF-CF"].astype(float)],
        axis=0,
        ignore_index=True,
    ).to_csv(output_dir / "y_test_unnorm.csv", index=False, header=False)

    a_train = df[["school_treatment", "occ_treatment"]].astype(int)
    for col in ["school_treatment", "occ_treatment"]:
        values = set(a_train[col].unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"{col} must be binary, got {sorted(values)}.")
    a_test = pd.concat(
        [
            a_train.assign(school_treatment=1 - a_train["school_treatment"]),
            a_train.assign(occ_treatment=1 - a_train["occ_treatment"]),
            a_train.assign(
                school_treatment=1 - a_train["school_treatment"],
                occ_treatment=1 - a_train["occ_treatment"],
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    a_train.to_csv(output_dir / "a_train.csv", index=False, header=False)
    a_test.to_csv(output_dir / "a_test.csv", index=False, header=False)

    base = build_base_covariates(df)
    serm_f = df[MULTI_SERM_FACTUAL].astype(float).copy()
    serm_f.columns = ["S", "E", "R", "M"]
    x_train = pd.concat([base, serm_f], axis=1)

    x_tests = []
    for suffix in MULTI_COUNTERFACTUALS:
        cols = [f"susceptible_{suffix}", f"exposed_{suffix}", f"recovered_{suffix}", f"dead_{suffix}"]
        serm_cf = df[cols].astype(float).copy()
        serm_cf.columns = ["S", "E", "R", "M"]
        x_tests.append(pd.concat([base, serm_cf], axis=1))
    x_test = pd.concat(x_tests, axis=0, ignore_index=True)
    x_train.to_csv(output_dir / "x_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)

    repeat_for_multi_training(output_dir)
    write_xa_files(output_dir, policy_dim=2)
    write_stats(output_dir, stats)
    return stats


def repeat_for_multi_training(output_dir: Path) -> None:
    for filename, has_header in [
        ("y_train.csv", False),
        ("x_train.csv", True),
        ("a_train.csv", False),
        ("y_train_unnorm.csv", False),
    ]:
        header = 0 if has_header else None
        df = pd.read_csv(output_dir / filename, header=header)
        repeated = pd.concat([df, df, df], axis=0, ignore_index=True)
        stem = Path(filename).stem
        repeated.to_csv(output_dir / f"{stem}_3.csv", index=False, header=has_header)


def write_xa_files(output_dir: Path, *, policy_dim: int) -> None:
    a_train = pd.read_csv(output_dir / "a_train.csv", header=None).astype(float)
    a_test = pd.read_csv(output_dir / "a_test.csv", header=None).astype(float)
    if a_train.shape[1] != policy_dim or a_test.shape[1] != policy_dim:
        raise ValueError("Unexpected policy dimension while writing xa files.")

    x_train = pd.read_csv(output_dir / "x_train.csv")
    x_test = pd.read_csv(output_dir / "x_test.csv")
    xa_train = pd.concat([a_train, x_train], axis=1)
    xa_test = pd.concat([a_test, x_test], axis=1)
    xa_train.to_csv(output_dir / "xa_train.csv", index=False, header=False)
    xa_test.to_csv(output_dir / "xa_test.csv", index=False, header=False)
    pd.Series(xa_train.to_numpy().reshape(-1)).to_csv(
        output_dir / "xa_train_flat.csv", index=False, header=False
    )
    pd.Series(xa_test.to_numpy().reshape(-1)).to_csv(
        output_dir / "xa_test_flat.csv", index=False, header=False
    )


def write_stats(output_dir: Path, stats: OutcomeStats) -> None:
    (output_dir / "outcome_stats.json").write_text(
        json.dumps(asdict(stats), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ABM output for ML baselines.")
    parser.add_argument("--setting", choices=["single", "multi"], required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.setting == "single":
        stats = preprocess_single(args.input_csv, args.output_dir)
    else:
        stats = preprocess_multi(args.input_csv, args.output_dir)
    print(f"Saved processed files to {args.output_dir}")
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
