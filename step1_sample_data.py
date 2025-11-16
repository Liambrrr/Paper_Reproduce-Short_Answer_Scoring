#!/usr/bin/env python3
"""
Sample the data and prepare per-question CSVs.

- Input:  train.tsv  (TSV file with columns: Id, EssaySet, Score1, Score2, EssayText)
- Output: answer/Q1.csv, ..., answer/Q10.csv
          Each CSV has columns: Id, EssaySet, EssayText, EssayScore
          where EssayScore is taken from Score1.

Sampling procedure (per question / EssaySet):
1. Randomly shuffle all responses for that question.
2. Take an initial sample of 100 responses.
3. While any score label (Score1 value) for that question appears
   fewer than MIN_FREQ times in the sample, add more responses of
   that label from the remaining pool (if available).
   Stop if all labels reach MIN_FREQ or if we run out of candidates.

Note: The total number of sampled responses may differ from the 1538
reported in the paper, since that depends on the random seed and the
exact dataset. This script follows the algorithmic description
(random 100, then add until label frequency ≥ 10).
"""

import os
import pandas as pd

INPUT_PATH = "train.tsv"
OUTPUT_DIR = "answer"
BASE_N = 100
MIN_FREQ = 10
RANDOM_SEED = 42


def sample_for_question(df_q, base_n=BASE_N, min_freq=MIN_FREQ, seed=RANDOM_SEED):
    """
    Given all rows for a single question (one EssaySet),
    return a sampled subset following:

    1. Randomly shuffle the rows.
    2. Start with the first `base_n` rows.
    3. Add more rows until each score label (Score1) for this question
       appears at least `min_freq` times in the selected subset,
       or until we run out of candidates.
    """
    df_q = df_q.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_q)

    if n <= base_n:
        return df_q.copy()

    selected_mask = [False] * n
    for i in range(base_n):
        selected_mask[i] = True

    total_counts = df_q["Score1"].value_counts().to_dict()
    all_labels = list(total_counts.keys())

    infeasible_labels = {
        label for label, count in total_counts.items() if count < min_freq
    }

    while True:
        selected = df_q[selected_mask]
        counts = selected["Score1"].value_counts().to_dict()

        need_more = []
        for label in all_labels:
            if label in infeasible_labels:
                continue
            current = counts.get(label, 0)
            if current < min_freq:
                need_more.append(label)

        if not need_more:
            break

        added_any = False

        for label in need_more:
            while True:
                current_count = selected["Score1"].value_counts().to_dict().get(label, 0)
                if current_count >= min_freq:
                    break

                found_candidate = False
                for idx in range(n):
                    if not selected_mask[idx] and df_q.at[idx, "Score1"] == label:
                        selected_mask[idx] = True
                        found_candidate = True
                        added_any = True
                        break

                if not found_candidate:
                    break

                selected = df_q[selected_mask]

        if not added_any:
            break

    return df_q[selected_mask]


def main():
    df = pd.read_csv(INPUT_PATH, sep="\t")
    required_cols = {"Id", "EssaySet", "Score1", "EssayText"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_selected = 0

    for q in sorted(df["EssaySet"].unique()):
        df_q = df[df["EssaySet"] == q]

        sampled_q = sample_for_question(df_q)

        out_q = sampled_q[["Id", "EssaySet", "EssayText", "Score1"]].copy()
        out_q = out_q.rename(columns={"Score1": "EssayScore"})

        out_path = os.path.join(OUTPUT_DIR, f"Q{q}.csv")
        out_q.to_csv(out_path, index=False)

        total_selected += len(out_q)

        print(f"Q{q}: selected {len(out_q)} responses")
        print(out_q["EssayScore"].value_counts().sort_index(), "\n")

    print(f"Total selected responses across all questions: {total_selected}")


if __name__ == "__main__":
    main()