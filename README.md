# llm-mp3-perf-test

Benchmarks local LLM inference speed and rating quality across models and machines, for the task of rating songs in a personal music library (1–100, structured output). Originally built to choose the best model/hardware combination for enriching a 10,531-song library — see [RESULTS.md](RESULTS.md) for findings and model recommendations.

The repo is self-contained: it ships with `songs_export.jsonl` (10,531 deduplicated songs — artist/title/album/genre only, no audio) so it can be cloned straight onto any machine (a fresh Azure VM, a RunPod GPU box, etc.) and run with no database or external setup beyond Ollama.

---

## Quick start

I'm using Ubuntu24 or 26 (on WSL2 and Proxmox)

```bash
git clone https://github.com/djhmateer/llm-mp3-perf-test.git
cd llm-mp3-perf-test
# so that uv is on PATH in this shell
source ./setup.sh

# winner — see RESULTS.md
ollama pull qwen3.6:35b
uv run bench.py --force-batch-size 4 --songs 4 --models qwen3.6:35b
uv run bench.py --force-batch-size 4 --songs 15000 --models qwen3.6:35b
```

`setup.sh` installs `uv` and Ollama (no models — pull whichever you want to test).

## Running the benchmark

```bash
# to load uv in a new shell
source $HOME/.local/bin/env

# smoke test
ollama pull qwen3:8b # 5GB
uv run bench.py --force-batch-size 4 --songs 4 --models qwen3:8b
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3:8b

# (36 songs — good for model comparison) - this is the winner model
ollama pull qwen3.6:35b
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3.6:35b

# runner up on CPU - good for speed
ollama pull qwen3:30b-instruct
uv run bench.py --force-batch-size 4 --songs 36 --models qwen3:30b-instruct
```

Larger models tested and **eliminated** — see [RESULTS.md](RESULTS.md) before pulling these, they're expensive and didn't beat the winner:

```bash
ollama pull qwen3.5:122b-a10b  # needs an A100 (80GB) — eliminated, genuine recall gap
ollama pull qwen3:235b-a22b    # needs a B200 (180GB) — eliminated, format-adherence bug
ollama pull qwen2.5:72b        # dense model — needs full GPU residency (A100+), don't use a 48GB card
```

Each batch prints a live progress line with an ETA for the whole run, e.g.:

```
songs 4045–4048  4/4 rated  0.60s/song  tps=45.9
songs 4049–4052  4/4 rated  0.59s/song  tps=47.2  eta=15:10:49
```

Logs are written to `logs/bench_<timestamp>.log`. Results are written to `results_<machine>_<timestamp>.csv`.

### Pulling results off a RunPod machine

`./download-logs.sh` fetches `results_*.csv` and `logs/*.log` from a remote box over `scp` (not `sftp` — Runpod's proxy doesn't support the sftp subsystem). It needs the pod's **direct TCP SSH connection** (not the `ssh.runpod.io` proxy string, which only allows interactive sessions):

```bash
./download-logs.sh
# prompts for: ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

Files land in `downloaded/<host>_<port>/`.

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
