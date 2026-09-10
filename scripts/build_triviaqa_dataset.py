#!/usr/bin/env python3
"""Build the TriviaQA question set (paper Appendix A.3).

Uses the validation split of the rc.nocontext configuration, which provides
question-answer pairs without the accompanying reading-comprehension passages,
since the paper tests recall rather than reading. Each question carries a
canonical answer and a list of aliases giving acceptable surface forms.

No GPU. Output: data/triviaqa.json
"""

import argparse
import json
import os

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                        "triviaqa.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="mandarjoshi/trivia_qa")
    parser.add_argument("--config", default="rc.nocontext")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset(args.dataset, args.config, split=args.split)
    data = []
    for row in rows:
        answer = row["answer"]
        data.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": answer["value"],
            "aliases": answer["aliases"],
            "normalized_answer": answer["normalized_value"],
            "normalized_aliases": answer["normalized_aliases"],
            "answer_type": answer["type"],
            "source": row["question_source"],
        })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} questions -> {OUT_FILE}")


if __name__ == "__main__":
    main()
