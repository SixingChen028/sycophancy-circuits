"""Shared Anthropic API helper for the two dataset-construction stages.

Both the description generator and the TriviaQA wrong-answer judge ask a model
for a small JSON object, in parallel, with retries. Neither touches a GPU.

Set ANTHROPIC_API_KEY in the environment before running either.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

__all__ = ["client", "ask_json", "run_parallel"]


def client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running this script.")
    return anthropic.Anthropic()


def ask_json(api, model, system, user, max_tokens=300, max_retries=5):
    """Ask for a JSON object and parse it, retrying with a backoff.

    Returns None if every attempt fails, so one bad item cannot abort a long run.
    """
    for attempt in range(max_retries):
        try:
            message = api.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # A model may emit other block types before the text, so join the
            # text blocks rather than indexing the first one.
            text = "".join(
                getattr(b, "text", "") for b in message.content
                if getattr(b, "type", None) == "text"
            ).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON object in response: {text[:120]}")
            return json.loads(match.group(0))
        except Exception as exc:
            wait = min(2 ** attempt, 30)
            print(f"  retry {attempt + 1} ({type(exc).__name__}: "
                  f"{str(exc)[:80]}) sleeping {wait}s", flush=True)
            time.sleep(wait)
    return None


def run_parallel(items, worker, workers, label="items"):
    """Map `worker` over `items` in a thread pool, dropping failures."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                results.append(result)
            if i % 100 == 0 or i == len(items):
                print(f"  {i}/{len(items)} {label}", flush=True)
    return results
