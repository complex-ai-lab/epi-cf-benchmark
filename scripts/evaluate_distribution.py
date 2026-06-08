import argparse
from pathlib import Path

import pandas as pd

from epicf_benchmark.evaluate import _load_array, evaluate_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sampled trajectory predictions.")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--num-units", type=int, default=158)
    parser.add_argument("--samples-per-unit", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = evaluate_distribution(
        _load_array(args.truth),
        _load_array(args.pred),
        num_units=args.num_units,
        samples_per_unit=args.samples_per_unit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
