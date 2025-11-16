"""
Step 6 - Evaluation

For each question Q1..Q10:
  - Load: number_result/Q{q}_NumberResult.csv
  - Columns: Id, EssaySet, EssayText, EssayScore,
             meta.llama3-1-405b-instruct-v1:0, NumberResult
  - Use EssayScore as ground-truth label, NumberResult as prediction.
  - Ignore rows where NumberResult == 'NA' (or is missing).
  - Compute:
      * Accuracy
      * Quadratic Weighted Kappa (QWK)
  - Compute averages across all 10 questions.
  - Save an additional report:
        evaluation_summary_llama_avg.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

NUMBER_RESULT_DIR = Path("number_result")
SUMMARY_PATH = Path("evaluation_summary_llama.csv")
SUMMARY_AVG_PATH = Path("evaluation_summary_llama_avg.csv")

QUESTIONS = range(1, 11)


def evaluate_question(q: int):
    filename = NUMBER_RESULT_DIR / f"Q{q}_NumberResult.csv"
    if not filename.exists():
        raise FileNotFoundError(f"Missing: {filename}")

    df = pd.read_csv(filename)

    for col in ["EssayScore", "NumberResult"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {filename}")

    n_total = len(df)

    # 1) Filter out explicit 'NA' / empty
    preds_raw = df["NumberResult"].astype(str).str.strip()
    mask_valid = (preds_raw.str.upper() != "NA") & (preds_raw != "") & (~preds_raw.isna())
    df_valid = df[mask_valid].copy()

    # 2) Convert NumberResult to numeric (handles '3', '3.0', '2.7', etc.)
    preds_num = pd.to_numeric(df_valid["NumberResult"], errors="coerce")

    # 3) Drop anything that still isn't numeric (coerce -> NaN)
    mask_numeric = ~preds_num.isna()
    df_valid = df_valid[mask_numeric].copy()
    preds_num = preds_num[mask_numeric]

    n_used = len(df_valid)

    if n_used == 0:
        return {
            "Question": f"Q{q}",
            "n_total": n_total,
            "n_used": n_used,
            "accuracy": np.nan,
            "qwk": np.nan,
        }

    # 4) Ground truth labels (EssayScore)
    y_true = df_valid["EssayScore"].astype(int).to_numpy()

    # 5) Predictions: round float to nearest int, then cast
    #    (if you prefer floor, use np.floor instead of round)
    y_pred = preds_num.round().astype(int).to_numpy()

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # QWK with quadratic weights
    all_labels = np.union1d(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=all_labels)

    return {
        "Question": f"Q{q}",
        "n_total": n_total,
        "n_used": n_used,
        "accuracy": acc,
        "qwk": qwk,
    }

def main():
    results = []

    for q in QUESTIONS:
        print(f"Evaluating Q{q}...")
        metrics = evaluate_question(q)
        results.append(metrics)

        print(
            f"  n_total = {metrics['n_total']}, "
            f"n_used = {metrics['n_used']}, "
            f"accuracy = {metrics['accuracy']:.4f}"
            if not np.isnan(metrics['accuracy']) else
            f"  n_total = {metrics['n_total']}, n_used = {metrics['n_used']}, accuracy = NAN"
        )

        print(
            f"  QWK = {metrics['qwk']:.4f}\n"
            if not np.isnan(metrics['qwk']) else
            "  QWK = NAN\n"
        )

    # Convert to DataFrame and save per-question results
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nPer-question summary saved to {SUMMARY_PATH.resolve()}")

    # ------------------------------
    # Compute averages across Q1–Q10
    # ------------------------------
    avg_accuracy = df_summary["accuracy"].mean(skipna=True)
    avg_qwk = df_summary["qwk"].mean(skipna=True)
    avg_valid_fraction = (df_summary["n_used"] / df_summary["n_total"]).mean()

    df_avg = pd.DataFrame([{
        "avg_accuracy": avg_accuracy,
        "avg_qwk": avg_qwk,
        "avg_valid_fraction": avg_valid_fraction
    }])

    df_avg.to_csv(SUMMARY_AVG_PATH, index=False)

    print("\n=== Average across all 10 questions ===")
    print(f"Average Accuracy      : {avg_accuracy:.4f}")
    print(f"Average QWK           : {avg_qwk:.4f}")
    print(f"Average Valid Fraction: {avg_valid_fraction:.4f}")
    print(f"Saved to {SUMMARY_AVG_PATH.resolve()}")


if __name__ == "__main__":
    main()