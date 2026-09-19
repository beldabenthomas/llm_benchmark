#!/usr/bin/env python3
"""
LLM latency & cost benchmark for auto-repair customer-service prompts.

For every (prompt, model) pair it streams a response and records:
  * TTFT        - time to first token (seconds from request start to first text chunk)
  * total time  - seconds until the stream finishes
  * tokens      - input/output token counts as reported by the provider
  * cost        - tokens x per-1M-token price from models.json
  * accuracy    - regex checks from prompts.json against the shop's price sheet

Usage:
  python benchmark.py                      # live run against the models in models.json
  python benchmark.py --runs 3             # repeat each prompt 3x (steadier medians)
  python benchmark.py --update-readme      # also write the results tables into README.md
  python benchmark.py --dry-run            # offline smoke test with simulated models
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

try:  # optional: load API keys from a local .env file
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    text: str
    ttft_s: float | None
    total_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class CallResult:
    model: str
    prompt_id: str
    run: int
    ttft_s: float | None
    total_s: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    checks_passed: int
    checks_total: int
    error: str
    output: str


# --------------------------------------------------------------------------- #
# Providers. Each takes (model_cfg, system_prompt, case) and returns StreamResult
# --------------------------------------------------------------------------- #
_clients: dict = {}


def _client(name, factory):
    if name not in _clients:
        _clients[name] = factory()
    return _clients[name]


def stream_openai(cfg: dict, system: str, case: dict) -> StreamResult:
    from openai import OpenAI

    client = _client("openai", OpenAI)
    parts: list[str] = []
    ttft = None
    in_tok = out_tok = 0

    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": case["prompt"]},
        ],
        stream=True,
        stream_options={"include_usage": True},
        **cfg.get("params", {}),
    )
    for chunk in stream:
        if chunk.usage:  # final chunk carries usage
            in_tok, out_tok = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            parts.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - start
    return StreamResult("".join(parts), ttft, total, in_tok, out_tok)


def stream_gemini(cfg: dict, system: str, case: dict) -> StreamResult:
    from google import genai
    from google.genai import types

    client = _client("gemini", genai.Client)  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    config = types.GenerateContentConfig(system_instruction=system, **cfg.get("params", {}))
    parts: list[str] = []
    ttft = None
    in_tok = out_tok = 0

    start = time.perf_counter()
    for chunk in client.models.generate_content_stream(
        model=cfg["model"], contents=case["prompt"], config=config
    ):
        if chunk.text:
            if ttft is None:
                ttft = time.perf_counter() - start
            parts.append(chunk.text)
        usage = chunk.usage_metadata
        if usage:
            in_tok = usage.prompt_token_count or 0
            # "thinking" tokens are billed as output tokens
            out_tok = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    total = time.perf_counter() - start
    return StreamResult("".join(parts), ttft, total, in_tok, out_tok)


def stream_mock(cfg: dict, system: str, case: dict) -> StreamResult:
    """Offline stand-in used by --dry-run to exercise the whole pipeline."""
    start = time.perf_counter()
    time.sleep(random.uniform(*cfg["mock_ttft_s"]))
    ttft = time.perf_counter() - start
    text = case.get("reference", "Hi! Yes, we are open today.")
    if random.random() > cfg["mock_accuracy"]:
        text = "I'm not sure about that, sorry."
    time.sleep(len(text.split()) * cfg["mock_s_per_word"])
    total = time.perf_counter() - start
    in_tok = int((len(system.split()) + len(case["prompt"].split())) * 1.3)
    return StreamResult(text, ttft, total, in_tok, int(len(text.split()) * 1.3))


PROVIDERS = {"openai": stream_openai, "gemini": stream_gemini, "mock": stream_mock}
REQUIRED_ENV = {"openai": ["OPENAI_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]}

MOCK_MODELS = [
    {"label": "Mock fast", "provider": "mock", "model": "mock-fast",
     "input_per_1m_usd": 0.10, "output_per_1m_usd": 0.40,
     "mock_ttft_s": [0.05, 0.15], "mock_s_per_word": 0.002, "mock_accuracy": 0.9},
    {"label": "Mock large", "provider": "mock", "model": "mock-large",
     "input_per_1m_usd": 2.50, "output_per_1m_usd": 10.00,
     "mock_ttft_s": [0.20, 0.40], "mock_s_per_word": 0.004, "mock_accuracy": 1.0},
]


# --------------------------------------------------------------------------- #
# Scoring and running
# --------------------------------------------------------------------------- #
def score(text: str, case: dict) -> tuple[int, int]:
    """Return (checks_passed, checks_total) using the regex rubric in prompts.json."""
    must = case.get("must_match", [])
    must_not = case.get("must_not_match", [])
    passed = sum(bool(re.search(p, text, re.I)) for p in must)
    passed += sum(not re.search(p, text, re.I) for p in must_not)
    return passed, len(must) + len(must_not)


RETRY_HINTS = ("429", "500", "502", "503", "504", "resource_exhausted", "unavailable",
               "overloaded", "rate limit", "quota")


def _is_retryable(message: str) -> bool:
    return any(h in message.lower() for h in RETRY_HINTS)


def run_call(cfg: dict, system: str, case: dict, run_idx: int, retries: int = 2) -> CallResult:
    """One timed call. Rate-limit/overload errors are retried with backoff; only the
    successful attempt is timed, so waiting between attempts never inflates latency."""
    base = dict(model=cfg["label"], prompt_id=case["id"], run=run_idx)
    for attempt in range(retries + 1):
        try:
            res = PROVIDERS[cfg["provider"]](cfg, system, case)
            break
        except Exception as exc:  # keep going; record the failure
            message = f"{type(exc).__name__}: {exc}"
            if attempt < retries and _is_retryable(message):
                wait = 20 * 2 ** attempt  # 20s, then 40s
                print(f"\n  [{cfg['label']}] rate-limited or busy, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{retries}). Reason: {message[:200]}")
                time.sleep(wait)
                continue
            return CallResult(**base, ttft_s=None, total_s=None, input_tokens=0, output_tokens=0,
                              cost_usd=0.0, checks_passed=0, checks_total=0, error=message, output="")
    passed, total = score(res.text, case)
    cost = (res.input_tokens * cfg["input_per_1m_usd"]
            + res.output_tokens * cfg["output_per_1m_usd"]) / 1_000_000
    return CallResult(**base, ttft_s=res.ttft_s, total_s=res.total_s,
                      input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                      cost_usd=cost, checks_passed=passed, checks_total=total,
                      error="", output=res.text.strip())


def check_env(models: list[dict]) -> None:
    for m in models:
        if m["provider"] not in PROVIDERS:
            sys.exit(f"Unknown provider '{m['provider']}' for {m['label']}.")
        names = REQUIRED_ENV.get(m["provider"])
        if names and not any(os.getenv(n) for n in names):
            sys.exit(f"Missing API key for {m['label']}: set {' or '.join(names)} (see .env.example).")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def _avg(xs):
    return statistics.fmean(xs) if xs else float("nan")


def _fmt(value, template):
    return "n/a" if value != value else template.format(value)  # NaN check


def compute_stats(results: list[CallResult], label: str) -> dict:
    all_calls = [r for r in results if r.model == label]
    ok = [r for r in all_calls if not r.error]
    checks_total = sum(r.checks_total for r in ok)
    return {
        "ttft_med": _med([r.ttft_s for r in ok if r.ttft_s is not None]),
        "total_med": _med([r.total_s for r in ok if r.total_s is not None]),
        "in_avg": _avg([r.input_tokens for r in ok]),
        "out_avg": _avg([r.output_tokens for r in ok]),
        "cost_avg": _avg([r.cost_usd for r in ok]),
        "cost_1k": _avg([r.cost_usd for r in ok]) * 1000,
        "acc": sum(r.checks_passed for r in ok) / checks_total if checks_total else float("nan"),
        "strict": _avg([float(r.checks_passed == r.checks_total) for r in ok]),
        "failed": len(all_calls) - len(ok),
    }


def collect_tables(results: list[CallResult], models: list[dict], cases: list[dict]):
    """Build the table data once; render it as markdown (README) or plain text (terminal)."""
    labels = [m["label"] for m in models]
    stats = {label: compute_stats(results, label) for label in labels}

    # (row title, stats key, format, which value wins, scale)
    rows = [
        ("Median time to first token (TTFT)", "ttft_med", "{:.0f} ms", "min", 1000),
        ("Median total latency", "total_med", "{:.2f} s", "min", 1),
        ("Avg input tokens / request", "in_avg", "{:.0f}", None, 1),
        ("Avg output tokens / request", "out_avg", "{:.0f}", None, 1),
        ("Avg cost / request", "cost_avg", "${:.6f}", "min", 1),
        ("Est. cost / 1,000 requests", "cost_1k", "${:.2f}", "min", 1),
        ("Accuracy (rubric checks passed)", "acc", "{:.0%}", "max", 1),
        ("Fully correct answers", "strict", "{:.0%}", "max", 1),
        ("Failed calls", "failed", "{:.0f}", None, 1),
    ]

    metric_rows = []  # (title, [(text, is_best), ...])
    for title, key, template, better, scale in rows:
        vals = [stats[l][key] * scale for l in labels]
        finite = [v for v in vals if v == v]
        best = (min(finite) if better == "min" else max(finite)) if better and finite else None
        cells = [(_fmt(v, template), best is not None and len(set(finite)) > 1 and v == best)
                 for v in vals]
        metric_rows.append((title, cells))

    prompt_rows = []  # (prompt text, ["pass" | "partial" | "fail" | "n/a", ...])
    for case in cases:
        statuses = []
        for label in labels:
            rs = [r for r in results if r.model == label and r.prompt_id == case["id"] and not r.error]
            if not rs:
                statuses.append("n/a")
                continue
            frac = sum(r.checks_passed for r in rs) / max(sum(r.checks_total for r in rs), 1)
            statuses.append("pass" if frac == 1 else "fail" if frac == 0 else "partial")
        short = case["prompt"] if len(case["prompt"]) <= 62 else case["prompt"][:59] + "..."
        prompt_rows.append((short, statuses))
    return labels, metric_rows, prompt_rows


def _note(cases: list[dict], runs: int) -> str:
    return f"{len(cases)} prompts x {runs} run(s) per model, run on {date.today().isoformat()}"


def build_markdown(results, models, cases, runs) -> str:
    labels, metric_rows, prompt_rows = collect_tables(results, models, cases)
    icons = {"pass": "✅", "partial": "⚠️", "fail": "❌", "n/a": "n/a"}
    lines = [
        f"_{_note(cases, runs)}. Best value per row in **bold**._",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for title, cells in metric_rows:
        lines.append(f"| {title} | " + " | ".join(f"**{t}**" if b else t for t, b in cells) + " |")
    lines += ["", "**Per-prompt accuracy** (✅ all checks passed, ⚠️ partial, ❌ none)", "",
              "| Prompt | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for prompt, statuses in prompt_rows:
        lines.append(f"| {prompt} | " + " | ".join(icons[s] for s in statuses) + " |")
    return "\n".join(lines)


def _text_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    fmt = lambda r: "   ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip()
    return "\n".join([fmt(headers), "   ".join("-" * w for w in widths)] + [fmt(r) for r in rows])


def build_text(results, models, cases, runs) -> str:
    """Aligned plain-text summary for the terminal (no markdown syntax, ASCII only)."""
    labels, metric_rows, prompt_rows = collect_tables(results, models, cases)
    metrics = [[t] + [f"{x} *" if b else x for x, b in cells] for t, cells in metric_rows]
    words = {"pass": "PASS", "partial": "PARTIAL", "fail": "FAIL", "n/a": "n/a"}
    prompts = [[p] + [words[s] for s in statuses] for p, statuses in prompt_rows]
    return "\n".join([
        f"RESULTS  ({_note(cases, runs)})",
        "",
        _text_table(["Metric"] + labels, metrics),
        "",
        "* = best value in that row",
        "",
        "ACCURACY BY PROMPT",
        "",
        _text_table(["Prompt"] + labels, prompts),
    ])


def write_csv(results: list[CallResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def update_readme(markdown: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)"
    if not re.search(pattern, text, flags=re.S):
        sys.exit("README.md is missing the RESULTS:START / RESULTS:END markers.")
    new = re.sub(pattern, lambda m: f"{m.group(1)}\n{markdown}\n{m.group(2)}", text, flags=re.S)
    path.write_text(new, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", default=str(ROOT / "models.json"))
    ap.add_argument("--prompts", default=str(ROOT / "prompts.json"))
    ap.add_argument("--runs", type=int, default=1, help="repeats per prompt (default 1)")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="min seconds between calls to the same model (free tiers: try 15)")
    ap.add_argument("--allow-failures", action="store_true",
                    help="write summary/README even if some calls failed")
    ap.add_argument("--verbose", action="store_true", help="print one line per API call")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warmup call")
    ap.add_argument("--update-readme", action="store_true", help="write results tables into README.md")
    ap.add_argument("--dry-run", action="store_true", help="simulate models offline (no API calls, no files written)")
    args = ap.parse_args()

    if args.dry_run and args.update_readme:
        sys.exit("--update-readme cannot be combined with --dry-run (simulated numbers).")

    models = MOCK_MODELS if args.dry_run else json.loads(Path(args.models).read_text())["models"]
    prompts = json.loads(Path(args.prompts).read_text())
    system, cases = prompts["system"], prompts["cases"]
    check_env(models)

    if not args.no_warmup:  # first request pays connection/TLS setup; keep it out of the numbers
        print("Warming up connections...")
        for m in models:
            w = run_call(m, system, {"id": "warmup", "prompt": "Hi, are you open today?"}, 0)
            if w.error:  # fail fast instead of burning through every prompt
                sys.exit(f"\nWarmup call to {m['label']} ({m['model']}) failed:\n  {w.error}\n\n"
                         "Check the API key and that the model name in your models file is still "
                         "available (providers retire models often).")

    results: list[CallResult] = []
    total_calls = args.runs * len(cases) * len(models)
    last_call: dict[str, float] = {}
    streak: dict[str, int] = collections.defaultdict(int)
    for run_idx in range(1, args.runs + 1):
        for case in cases:
            for m in models:  # interleave models so network drift hits both equally
                wait = args.min_interval - (time.monotonic() - last_call.get(m["label"], -1e9))
                if wait > 0:
                    time.sleep(wait)  # pacing happens outside the timed region
                r = run_call(m, system, case, run_idx)
                last_call[m["label"]] = time.monotonic()
                results.append(r)
                streak[m["label"]] = streak[m["label"]] + 1 if r.error else 0
                if r.error:
                    print(f"\n[{m['label']}] {case['id']} run{run_idx}: ERROR {r.error[:300]}")
                elif args.verbose:
                    ttft = f"{r.ttft_s * 1000:.0f} ms" if r.ttft_s is not None else "n/a"
                    print(f"[{m['label']}] {case['id']} run{run_idx}: TTFT {ttft}, "
                          f"total {r.total_s:.2f}s, {r.output_tokens} out tok, "
                          f"{r.checks_passed}/{r.checks_total} checks")
                if streak[m["label"]] >= 4 and not args.dry_run:
                    sys.exit(f"\nStopping: 4 calls in a row failed for {m['label']}. This is usually a "
                             "rate or daily quota limit. Nothing was saved.\nWait a few minutes and retry "
                             "with e.g. --min-interval 15, or use fewer --runs.")
                if not args.verbose:
                    print(f"\r  {len(results)}/{total_calls} calls complete", end="", flush=True)
    if not args.verbose:
        print()

    markdown = build_markdown(results, models, cases, args.runs)
    print("\n" + build_text(results, models, cases, args.runs))
    print(f"\nTotal spend for this run: ${sum(r.cost_usd for r in results):.4f}")

    failed = [r for r in results if r.error]
    if failed:
        print(f"\nWARNING: {len(failed)} of {len(results)} calls failed, so the table above is "
              "based on fewer samples than intended. Most common errors:")
        for msg, n in collections.Counter(r.error[:160] for r in failed).most_common(3):
            print(f"  {n}x {msg}")

    dead = [m["label"] for m in models
            if not any(r.model == m["label"] and not r.error for r in results)]
    if dead and not args.dry_run:  # never overwrite results/README with an all-failed run
        sys.exit(f"\nNo successful calls for: {', '.join(dead)}. Nothing was saved and README.md was left untouched.")

    if not args.dry_run:
        RESULTS_DIR.mkdir(exist_ok=True)
        write_csv(results, RESULTS_DIR / "raw_results.csv")
        print(f"Wrote {RESULTS_DIR / 'raw_results.csv'}")
        if failed and not args.allow_failures:
            print("Skipped summary.md and README.md because of the failed calls. Fix the errors and "
                  "re-run (or pass --allow-failures to save the partial results anyway).")
            return
        (RESULTS_DIR / "summary.md").write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {RESULTS_DIR / 'summary.md'}")
        if args.update_readme:
            update_readme(markdown, ROOT / "README.md")
            print("Updated README.md")


if __name__ == "__main__":
    main()
