# llm-mp3-perf-test

Benchmarks local LLM inference speed and rating quality across models and machines, for the task of rating songs in a personal music library (1–100, structured output). Originally built to choose the best model/hardware combination for enriching a 10,531-song library — see [RESULTS.md](RESULTS.md) for findings and model recommendations.

The repo is self-contained: it ships with `songs_export.jsonl` (10,531 deduplicated songs — artist/title/album/genre only, no audio) so it can be cloned straight onto any machine (a fresh Azure VM, a RunPod GPU box, etc.) and run with no database or external setup beyond Ollama.

---

## Quick start

```bash
git clone <this-repo-url>
cd llm-mp3-perf-test
chmod +x setup.sh && ./setup.sh
uv run bench.py --force-batch-size 4 --machine <your-machine-name>
```

`setup.sh` installs `uv`, installs Ollama if missing, and pulls the current leaderboard models (see [RESULTS.md](RESULTS.md)). Edit the `MODELS` array in `setup.sh` to pull different models.

---

## Running the benchmark

```bash
# Full run — Part 1 sweeps batch sizes 1–20 to find optimal, then Part 2 benchmarks
uv run bench.py

# Skip Part 1 if you already know the optimal batch size (4 on CPU — see RESULTS.md)
uv run bench.py --force-batch-size 4

# Recommended: 100 songs gives reliable quality stats, ~36 min on CPU per model
uv run bench.py --force-batch-size 4 --songs 100

# Quick smoke test (36 songs — good for model comparison)
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3.6:35b

# Single model
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3:30b-instruct

# Tiny smoke test (4 songs) — confirm a model runs before a full benchmark
uv run bench.py --force-batch-size 4 --songs 4 --models qwen3.6:35b

# Tag results with a machine name (recommended when comparing across hosts)
uv run bench.py --force-batch-size 4 --machine a100-runpod
```

### How many songs?

| Songs | Warm batches | Rated values | CPU time/model | GPU time/model (est.) |
|-------|-------------|--------------|----------------|-----------------------|
| 20 | ~4 | ~14 | 7 min | <1 min |
| 50 | ~12 | ~35 | 18 min | ~2 min |
| **100** | **~24** | **~70** | **~36 min** | **~4 min** |

**100 is the recommended default** — 70 rated values gives meaningful std dev and distribution stats.

Logs are written to `logs/bench_<timestamp>.log`. Results to `results_<machine>_<timestamp>.csv`.

### Fallback (no uv)

```bash
pip install ollama pydantic
python bench.py --force-batch-size 4 --songs 100
```

---

## Metrics reference

| Metric | Description | Target |
|--------|-------------|--------|
| Tokens/sec (TPS) | Generation throughput — main speed metric | — |
| Time to first token (TTFT) | Latency before output starts | — |
| s/song | Wall-clock time per song | — |
| Valid rating % | % returning a parseable integer 1–100 | 100% |
| Null rate % | % where model doesn't recognise the song | <20% |
| Rating mean | Average rating across all rated songs | ~65 |
| Rating std dev | Spread — higher means model uses the full scale | >10 |
| Rated 90+ % | % of songs rated 90 or above | ~5% |

See [RESULTS.md → What the Metrics Mean](RESULTS.md#what-the-metrics-mean) for a worked explanation of std dev and mean.

## Output

One CSV per run: `results_<machine>_<timestamp>.csv` — all models combined, sorted by rating descending. Everything also logged to `logs/bench_<timestamp>.log`.

These are gitignored — they're run artifacts, not part of the repo. Send the CSV + log back to whoever's collating results across machines.
