#!/usr/bin/env python3
"""Build the MMLU STEM question set (paper Appendix A.1, Table 1).

Draws 15 subjects spanning the natural sciences, medicine and engineering, and
excludes subjects that primarily require multi-step computation such as
mathematics and formal logic, since those are known to rely on mechanisms
distinct from factual retrieval.

Questions whose prompt exceeds 1,000 characters are excluded, leaving the 2,476
candidates the paper reports. That filter is applied during screening, so the
full set is written here and the summary below reports both counts; the
post-filter per-subject counts are the ones in Table 1.

No GPU. Output: data/mmlu_stem.json
"""

import argparse
import json
import os
from collections import Counter

# The 15 subjects of Table 1.
SUBJECTS = [
    "anatomy",
    "astronomy",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "medical_genetics",
    "virology",
]

# Screening drops anything longer than this; reported here to reproduce Table 1.
MAX_QUESTION_CHARS = 1000

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                        "mmlu_stem.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="cais/mmlu")
    args = parser.parse_args()

    from datasets import load_dataset

    data = []
    for subject in SUBJECTS:
        rows = load_dataset(args.dataset, subject, split=args.split)
        for row in rows:
            data.append({
                "question": row["question"],
                "choices": list(row["choices"]),
                "correct_answer": int(row["answer"]),
                "subject": subject,
            })
        print(f"  {len(rows):5d}  {subject}", flush=True)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(data)} questions -> {OUT_FILE}")

    kept = [e for e in data if len(e["question"]) <= MAX_QUESTION_CHARS]
    print(f"\nTable 1 (after subject and length filtering, "
          f"<= {MAX_QUESTION_CHARS} characters):")
    counts = Counter(e["subject"] for e in kept)
    for subject in SUBJECTS:
        print(f"  {subject.replace('_', ' ').capitalize():25s} {counts[subject]:5d}")
    print(f"  {'Total':25s} {len(kept):5d}")


if __name__ == "__main__":
    main()
