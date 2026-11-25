"""Generate an isolated Kaggle submission from the observed ID threshold."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--threshold", type=int, default=7500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/submission_id_le_7500.csv"),
    )
    args = parser.parse_args()

    test = pd.read_csv(args.test, dtype={"id": str})
    if "id" not in test.columns:
        parser.error(f"{args.test}: missing id column")
    if test["id"].eq("").any() or test["id"].duplicated().any():
        parser.error(f"{args.test}: IDs must be nonempty and unique")
    try:
        numeric_ids = pd.to_numeric(test["id"], errors="raise")
    except (TypeError, ValueError) as error:
        parser.error(f"{args.test}: IDs must be integers: {error}")

    submission = pd.DataFrame({
        "id": test["id"],
        "label": (numeric_ids <= args.threshold).astype(int),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    counts = submission["label"].value_counts().sort_index().to_dict()
    print(f"Saved {len(submission)} rows to {args.output}")
    print(f"Rule: label=1 when id <= {args.threshold}; otherwise label=0")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
