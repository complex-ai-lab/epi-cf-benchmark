import argparse
from pathlib import Path

import pandas as pd

from epicf_benchmark.evaluate import _load_array, evaluate_cate_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CATE RMSE from factual/counterfactual trajectories.")
    parser.add_argument("--factual-truth", required=True)
    parser.add_argument("--counterfactual-truth", required=True)
    parser.add_argument("--factual-pred", required=True)
    parser.add_argument("--counterfactual-pred", required=True)
    parser.add_argument("--num-units", type=int, default=158)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    value = evaluate_cate_rmse(
        _load_array(args.factual_truth),
        _load_array(args.counterfactual_truth),
        _load_array(args.factual_pred),
        _load_array(args.counterfactual_pred),
        num_units=args.num_units,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cate_rmse": value}]).to_csv(output, index=False)
    print(f"CATE RMSE: {value:.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
