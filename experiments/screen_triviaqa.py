#!/usr/bin/env python3
"""Four-stage screening for free-form TriviaQA (paper Appendix A.3).

Unlike MMLU there is no fixed option set, so the wrong answer the user states has
to be constructed, and every screen is evaluated on the model's generated text.

  --stage plain    Screen 1. The model's answer to the plain prompt matches the
                   gold answer or one of its aliases.

  --stage guess    Screen 2. Construct a wrong answer from the model's own
                   next-best guess: re-run the plain prompt with beam search and
                   take the highest-ranked beam whose normalized text differs
                   from every gold alias.

  --stage opinion  Screen 3. State that wrong answer as the user's opinion and
                   require the response to adopt it specifically -- matching the
                   stated answer, not merely being wrong.

  --stage build    Merge the stages into one screened_examples.json and apply
                   screen 4, which drops questions whose correct and wrong
                   answers share a first token. The readout compares the two
                   first tokens, so those questions carry no signal.

Between `guess` and `opinion`, run scripts/judge_triviaqa_answers.py: beam search
sometimes produces degenerate continuations that echo the question or run on
across clauses rather than giving a second candidate answer, and the judge filters
those out.

Output: runs/<model>/triviaqa/{plain_answers,second_guess,opinion_screen,
                               screened_examples}.json
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syco import prompts as P
from syco.data import first_token_id
from syco.triviaqa import gold_set, is_correct, is_sycophantic, normalize
from syco.runtime import base_parser, batches, load_model, run_path

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                       "triviaqa.json")
MAX_NEW_TOKENS = 16


def read_json(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. {hint}")
    with open(path) as f:
        return json.load(f)


def write_json(path, records, label):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} {label} -> {path}")


def generate(model, tok, prompts, device, batch_size, num_beams=1, label=""):
    """Greedy or beam-search continuations, returned as raw strings.

    With beam search, returns one list of `num_beams` strings per prompt.
    """
    outputs = []
    for start, batch in batches(prompts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        if num_beams > 1:
            outputs.extend(
                decoded[i:i + num_beams] for i in range(0, len(decoded), num_beams)
            )
        else:
            outputs.extend(decoded)
        print(f"  {label} [{start + len(batch)}/{len(prompts)}]", flush=True)
    return outputs


def stage_plain(args, spec, tok, model, device):
    data = read_json(DATASET, "Run scripts/build_triviaqa_dataset.py first.")
    if args.limit:
        data = data[:args.limit]
    prompts = [P.triviaqa_plain(tok, e["question"]) for e in data]
    answers = generate(model, tok, prompts, device, args.batch_size, 1, "plain")

    results = []
    for example, answer in zip(data, answers):
        results.append({
            **example,
            "plain_raw_answer": answer.strip(),
            "plain_normalized": normalize(answer),
            "screen1": bool(is_correct(answer, example)),
        })
    n = sum(r["screen1"] for r in results)
    print(f"\nScreen 1 (plain correct): {n}/{len(results)} ({n / len(results):.1%})")
    write_json(run_path(args.model, "triviaqa", "plain_answers.json"),
               results, "plain answers")


def stage_guess(args, spec, tok, model, device):
    """The wrong answer is the model's own next-best guess."""
    records = read_json(
        run_path(args.model, "triviaqa", "plain_answers.json", create=False),
        "Run --stage plain first.")
    pool = [r for r in records if r["screen1"]]
    if args.limit:
        pool = pool[:args.limit]
    print(f"Beam search over {len(pool)} plain-correct questions "
          f"({args.num_beams} beams).", flush=True)

    prompts = [P.triviaqa_plain(tok, e["question"]) for e in pool]
    beams = generate(model, tok, prompts, device, max(1, args.batch_size // 4),
                     args.num_beams, "beams")

    results = []
    for example, candidates in zip(pool, beams):
        gold = gold_set(example)
        # The highest-ranked beam that is not the gold answer.
        wrong = next((c.strip() for c in candidates
                      if normalize(c) and normalize(c) not in gold), None)
        results.append({
            **example,
            "beams": [c.strip() for c in candidates],
            "wrong_guess": wrong,
            "screen2": wrong is not None,
        })
    n = sum(r["screen2"] for r in results)
    print(f"\nScreen 2 (wrong answer found): {n}/{len(results)}")
    write_json(run_path(args.model, "triviaqa", "second_guess.json"),
               results, "second guesses")


def stage_opinion(args, spec, tok, model, device):
    path = run_path(args.model, "triviaqa", "second_guess_validated.json",
                    create=False)
    if not os.path.exists(path):
        path = run_path(args.model, "triviaqa", "second_guess.json", create=False)
        print("Note: using unvalidated second guesses. Run "
              "scripts/judge_triviaqa_answers.py to filter beam-search "
              "artifacts first.", flush=True)
    records = read_json(path, "Run --stage guess first.")
    pool = [r for r in records
            if r.get("screen2") and r.get("wrong_guess")
            and r.get("judge_valid", True)]
    if args.limit:
        pool = pool[:args.limit]
    print(f"Stating a wrong answer for {len(pool)} questions.", flush=True)

    prompts = [P.triviaqa_deceptive(tok, e["question"], e["wrong_guess"])
               for e in pool]
    answers = generate(model, tok, prompts, device, args.batch_size, 1, "opinion")

    results = []
    for example, answer in zip(pool, answers):
        results.append({
            **example,
            "dec_raw_answer": answer.strip(),
            "dec_normalized": normalize(answer),
            "dec_correct": bool(is_correct(answer, example)),
            "screen3": bool(is_sycophantic(answer, example["wrong_guess"], example)),
        })
    n = sum(r["screen3"] for r in results)
    print(f"\nScreen 3 (adopted the stated answer): {n}/{len(results)} "
          f"({n / max(len(results), 1):.1%})")
    write_json(run_path(args.model, "triviaqa", "opinion_screen.json"),
               results, "opinion screens")


def stage_build(args, spec, tok, model, device):
    """Merge the stages and apply screen 4 (distinct first tokens)."""
    records = read_json(
        run_path(args.model, "triviaqa", "opinion_screen.json", create=False),
        "Run --stage opinion first.")

    results = []
    for i, record in enumerate(records):
        correct_token = first_token_id(tok, record["plain_raw_answer"])
        wrong_token = first_token_id(tok, record["wrong_guess"])
        screen4 = correct_token != wrong_token
        results.append({
            **record,
            "orig_idx": i,
            "correct_first_token": int(correct_token),
            "wrong_first_token": int(wrong_token),
            "screen4": bool(screen4),
            "pass_all": bool(record["screen1"] and record["screen2"]
                             and record["screen3"] and screen4),
        })
    n = sum(r["pass_all"] for r in results)
    print(f"\nScreen 4 (distinct first tokens): "
          f"{sum(r['screen4'] for r in results)}/{len(results)}")
    print(f"Passing all screens: {n}/{len(results)}")
    write_json(run_path(args.model, "triviaqa", "screened_examples.json"),
               results, "screened examples")


STAGES = {
    "plain": stage_plain,
    "guess": stage_guess,
    "opinion": stage_opinion,
    "build": stage_build,
}


def main():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beams for the wrong-answer search (stage 'guess')")
    parser.set_defaults(batch_size=64)
    args = parser.parse_args()

    spec, tok, model, device = load_model(args.model, verify=False)
    STAGES[args.stage](args, spec, tok, model, device)


if __name__ == "__main__":
    main()
