# llm-mp3-perf-test

Benchmarks local LLM inference speed and rating quality across models and machines, for the task of rating songs in a personal music library (1–100, structured output). Originally built to choose the best model/hardware combination for enriching a 10,531-song library — see [RESULTS.md](RESULTS.md) for findings and model recommendations.

The repo is self-contained: it ships with `songs_export.jsonl` (10,531 deduplicated songs — artist/title/album/genre only, no audio) so it can be cloned straight onto any machine (a fresh Azure VM, a RunPod GPU box, etc.) and run with no database or external setup beyond Ollama.

---

## Quick start

I'm using Ubuntu24 or 26 (on WSL2 and Proxmox)

```bash
cd llm-mp3-perf-test
chmod +x setup.sh 
# so that uv will run in the 
source ./setup.sh
uv run bench.py --force-batch-size 4 
```

setup.sh installs `uv`, installs Ollama if missing, and pulls a small model - qwen3:8b 

## Running the benchmark

```bash
# to load uv
source $HOME/.local/bin/env

# smoke test - bad model (setup pulls this)
ollama pull qwen3:8b # 5GB
uv run bench.py --force-batch-size 4 --songs 4 --models qwen3:8b
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3:8b

# (36 songs — good for model comparison) - this is the winner model on CPU
ollama pull qwen3.6:35b
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3.6:35b

# runner up on CPU - good for speed
ollama pull qwen3:30b-instruct
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3:30b-instruct
```

Logs are written to `logs/bench_<timestamp>.log`. Results to `results_<machine>_<timestamp>.csv`.

Result is written to `results_<machine>_<timestamp>.csv`

### Fallback (no uv)

```bash
pip install ollama pydantic
python bench.py --force-batch-size 4 --songs 100
```

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
