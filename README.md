# LLM Latency & Cost Benchmark: Auto Repair Customer Service

A small, dependency-light Python script that compares two LLMs on the kind of questions an auto repair shop's virtual receptionist gets every day (*"How much is an oil change for a 2018 Honda Civic?"*).

For each prompt and model it measures:

| Metric | How it's measured |
|---|---|
| **Time to first token (TTFT)** | `time.perf_counter()` from just before the request is sent to the first non-empty streamed text chunk |
| **Total latency** | Same clock, until the stream ends |
| **Tokens & cost** | Token counts reported by the provider's API x per-1M-token prices in [`models.json`](models.json) |
| **Accuracy** | Regex rubric in [`prompts.json`](prompts.json) checking answers against the shop's price sheet |

## Results

<!-- RESULTS:START -->
_9 prompts x 3 run(s) per model, run on 2026-09-19. Best value per row in **bold**._

| Metric | Gemini 3.5 Flash-Lite | Gemini 3.1 Flash-Lite |
|---|---|---|
| Median time to first token (TTFT) | 697 ms | **571 ms** |
| Median total latency | 0.83 s | **0.71 s** |
| Avg input tokens / request | 276 | 276 |
| Avg output tokens / request | 54 | 51 |
| Avg cost / request | $0.000219 | **$0.000145** |
| Est. cost / 1,000 requests | $0.22 | **$0.15** |
| Accuracy (rubric checks passed) | 100% | 100% |
| Fully correct answers | 100% | 100% |
| Failed calls | 0 | 0 |

**Per-prompt accuracy** (✅ all checks passed, ⚠️ partial, ❌ none)

| Prompt | Gemini 3.5 Flash-Lite | Gemini 3.1 Flash-Lite |
|---|---|---|
| How much is an oil change for a 2018 Honda Civic? | ✅ | ✅ |
| My check engine light just came on. What will it cost to lo... | ✅ | ✅ |
| Are you open on Sundays? | ✅ | ✅ |
| How much for new brake pads on my 2016 Toyota Camry? | ✅ | ✅ |
| Do I need an appointment for a tire rotation? | ✅ | ✅ |
| My car won't start and I think it's the battery. Can you re... | ✅ | ✅ |
| Can you do a transmission rebuild for my 2010 Ford F-150? W... | ✅ | ✅ |
| My AC is blowing warm air. What's the price and how long do... | ✅ | ✅ |
| I'm on a budget. Can you do a tire rotation and an oil chan... | ✅ | ✅ |
<!-- RESULTS:END -->

**How to read it:** lower is better for latency and cost, higher is better for accuracy, and the best value in each row is in bold. TTFT is how long a user waits before text starts appearing, which matters most for chat and voice assistants.

**Models tested:** Gemini 3.5 Flash-Lite vs Gemini 3.1 Flash-Lite (config: [`models.gemini-lite-pair.json`](models.gemini-lite-pair.json)).

**Key takeaways**

- **Accuracy did not separate the models.** Both passed every rubric check (100%), so this test is too easy to rank quality. It mainly shows both are adequate for short answers grounded in a provided price sheet.
- **Gemini 3.1 Flash-Lite was faster and cheaper on this workload:** median time to first token 571 ms vs 697 ms (about 18% lower), and about $0.15 vs $0.22 per 1,000 requests (about a third cheaper, at list prices).
- **Why it's cheaper:** token counts were nearly identical (276 input, about 51-54 output), so the gap comes mostly from 3.1's lower per-token price rather than shorter answers.
- **Practical read:** for short customer-service replies grounded in supplied data, the cheaper model is enough. A harder prompt set (ambiguous requests, customers pushing for discounts, questions the price sheet doesn't cover) would be needed to find where a more expensive model earns its price.
- **Caveats:** 27 calls per model, one machine and network, free-tier API, costs computed from list prices checked in Sept 2026. Latency gaps of about 100 ms are indicative and worth re-checking at a different time of day.

Per-call data lives in [`results/raw_results.csv`](results/) after a run.

## Quick start

```bash
git clone https://github.com/beldabenthomas/llm_benchmark.git && cd llm_benchmark
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                # Windows: copy .env.example .env, then paste in your API key(s)

python benchmark.py --dry-run                       # offline smoke test, no keys needed
python benchmark.py --models models.gemini-lite-pair.json --runs 3 --min-interval 8 --update-readme   # reproduces the table above (Gemini key only)
python benchmark.py --verbose                       # optional: also print one line per API call
```

A full run of 9 prompts x 3 repeats x 2 models is 54 short requests, which cost about one cent at list prices (and $0 on Gemini's free tier).

## How it works

1. **Shop context.** A fictional shop ("Northside Auto Repair") has a small price sheet and hours in the system prompt, so every question has a checkable ground truth.
2. **Prompts.** 9 realistic customer questions, including edge cases: a repair that is *not* on the price sheet (the model should not invent a price), a multi-item total (arithmetic), and hours/walk-in policy.
3. **Streaming.** Both providers are called with streaming enabled so TTFT can be observed directly.
4. **Fairness.** One untimed warmup call per model absorbs connection setup; calls are sequential and interleaved (model A, model B, next prompt...) so network drift affects both; temperature is 0.
5. **Scoring.** Each prompt has `must_match` regexes (facts that must appear, e.g. `89\.99`) and `must_not_match` regexes (e.g. hallucinated four-digit prices). Accuracy is the share of checks passed; "fully correct" is the share of answers passing all checks.

## Configuration

**Model config files** (pick one with `--models`):

- [`models.gemini-lite-pair.json`](models.gemini-lite-pair.json): Gemini 3.5 vs 3.1 Flash-Lite. Used for the published results; needs only a Gemini key.
- [`models.gemini-only.json`](models.gemini-only.json): Gemini Flash-Lite vs the larger Flash. The larger model has a small free daily quota, so a full run may need a paid tier.
- [`models.json`](models.json): default; one OpenAI model and one Gemini model (needs both keys).

**Swap models or update prices** in any of these files. Example entry:

```json
{
  "label": "GPT-4o mini",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "input_per_1m_usd": 0.15,
  "output_per_1m_usd": 0.60,
  "params": { "temperature": 0, "max_completion_tokens": 300 }
}
```

`provider` is `openai` or `gemini`. `params` is passed straight to the provider SDK. Pricing changes often, so check each provider's pricing page before quoting the cost numbers.

**Add prompts** in [`prompts.json`](prompts.json): each case needs an `id`, a `prompt`, its `must_match` / `must_not_match` regexes, and a `reference` answer (used only by `--dry-run`).

## Caveats

- **Free-tier rate limits.** Free API tiers allow only a few requests per minute. If you see `429` errors, add `--min-interval 15` (seconds between calls to the same model). Rate-limited calls are retried with backoff, and waiting time is never counted in the latency numbers. If any call still fails, the script saves only the raw CSV and refuses to overwrite this README with partial results (override with `--allow-failures`).
- **Small sample.** 9 prompts is enough to see a difference in speed and cost, not to rank models on quality. Use `--runs` to smooth out latency noise.
- **Latency depends on where you run it.** Network path, time of day, and provider load all move TTFT. Compare models within one run, not across machines.
- **Reasoning ("thinking") models** spend time and output tokens before the first visible token. That counts toward TTFT and cost here, which is what a real user would experience.
- **The accuracy rubric is keyword-based.** It reliably catches wrong prices and hallucinated quotes but can't judge tone or phrasing. An LLM-as-judge step would be a natural next step.

## Extending to ASR

The same harness fits speech-to-text: replace `case["prompt"]` with an audio file, add a provider function (e.g. OpenAI Whisper / `gpt-4o-transcribe`), measure time-to-first-transcript-chunk, and score with word error rate against a reference transcript.

## Project layout

```
benchmark.py        # the benchmark script
models*.json        # model configs + pricing (see Configuration)
prompts.json        # shop context, prompts, accuracy rubric
requirements.txt
.env.example
results/            # raw_results.csv + summary.md from the published run
```
