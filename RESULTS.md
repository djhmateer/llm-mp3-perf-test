# Phase 9 — Benchmark Results

Rating quality and inference speed across models and machines, for the task of rating 10,531 songs. For how to run the benchmark yourself, see [README.md](README.md).

**Methodology:** Fixed random seed, batch size 4 (sweep-confirmed optimal on all machines), temperature 0.1, ollama library with Pydantic structured output (`format=`), `think=False` for thinking models. Quality metrics from 200-song runs where available.

**Quality targets:** std dev >10, null rate <20%, mean ~65, 90+% ~5%.

---

## What the Metrics Mean

### Mean rating
The average rating across all rated songs. Target is ~65, meaning an "average" song in the library should score around 65, leaving room to go higher for great songs and lower for weak ones.

When the mean is too high (e.g. qwen3:30b-instruct at 78.4), the model is too generous — almost every song sounds good to it, which squashes the useful signal at the top end. qwen3.6:35b at 71.7 is the closest to the target of any non-reasoning model tested.

### Std dev (standard deviation)
Measures how spread out the ratings are. A high std dev means the model is using the full scale and genuinely differentiating between songs. A low std dev means everything clusters around the mean — the model isn't picking winners.

Example from qwen3.6:35b (std dev 11.8):

| Song | Rating |
|------|--------|
| Julianne Hough / Tom Cruise - Rock You Like a Hurricane | 45 |
| Eels - P.S. You Rock My World | 60 |
| British Sea Power - We Close Our Eyes | 65 |
| Beck - Emergency Exit | 74 |
| Oasis - Slide Away | 78 |
| Janis Joplin - Piece Of My Heart | 85 |
| The Smiths - There Is a Light That Never Goes Out | 92 |

Spread from 45 to 92 — the model is confidently saying "this cover version is mediocre, this is decent, this is a genuine classic."

Compare phi4:14b (std dev 6.4): everything came back 65–82, with The Smiths at only 82. It can't tell the difference between a decent song and a classic. Useless for building a ranked playlist.

**Mean tells you if the model is calibrated. Std dev tells you if the model actually has opinions.**

---

## Winner: qwen3.6:35b

Best model tested on every quality metric. Runs on CPU DDR5 (64GB RAM).

**What it is:** A Mixture of Experts (MoE) model from Alibaba's Qwen 3.6 family. The full ollama tag is `qwen3.6:35b-a3b` — 35B *total* parameters, but only 3B *active* per token. At inference time, each token is routed to a small subset of specialist sub-networks ("experts") rather than flowing through all 35B parameters. This is why it runs at 7–8 TPS on CPU despite the 35B label — it's doing roughly the same compute per token as a 3B dense model, while drawing on knowledge spread across the full 35B. The Q4 quantization brings the weights down to ~22GB, comfortably within 64GB RAM.

| Metric | Value | Notes |
|--------|-------|-------|
| TPS | 7.9 | |
| s/song | 3.48s | ~10h for full 10,531-song library on DDR5 |
| Std dev | **11.8** | Best of all models — widest range of opinions, from 45 for a mediocre cover to 92 for a classic |
| Null rate | **8%** | Best of all models — knows 92% of the library |
| 90+% | **5%** | Exactly on target |
| Mean | **71.7** | Closest to target 65 of any non-reasoning model |
| Valid ratings | 100% | |

---

## Comparison (200-song runs, DDR5)

| Model | TPS | s/song | Std dev | Null% | 90+% | Mean | Full library est. | Status |
|-------|-----|--------|---------|-------|------|------|--------------------|--------|
| **qwen3.6:35b** | 7.9 | 3.48s | **11.8** | **8%** | **5%** | **71.7** | ~10.1h | **Winner** |
| qwen3:30b-instruct | 12.9 | 2.22s | 9.5 | 11% | 8% | 78.4 | ~6.5h | Runner-up (speed) |
| gemma4:26b | 6.9 | 4.03s | 10.2 | 23% | 10% | 76.5 | ~11.8h | Runner-up (quality) |
| mistral-small3.2:24b | 2.6 | 10.12s | 12.0 | 34% | 6% | 72.2 | ~29.6h | High spread, too many nulls |
| phi4:14b | 3.9 | 5.63s | 6.4 | 34% | 5% | 74.8 | — | Eliminated — terrible spread |
| deepseek-r1:32b | 2.0 | 59.77s | 9.8 | 25% | 0% | 66.6 | — | Eliminated — think=False ignored, 190s TTFT |

---

## All Models Tested

### Eliminated on WSL2 (DDR4, ~32GB)

| Model | Reason |
|-------|--------|
| qwen3:8b | Std dev 7.5, 28% null — too small |
| qwen3:14b | Std dev 10.3 but 31% null and slow (9.7s/song) |
| qwen3:32b | Same speed as qwen2.5:32b (1.1 TPS), worse quality — pointless |
| qwen2.5:72b | OOM (~45GB model, ~32GB RAM) |

### Eliminated on DDR5 (64GB)

| Model | Reason |
|-------|--------|
| qwen2.5:32b | 2.1 TPS, std dev 8.1, 36% null — superseded |
| llama3.3:70b | 38% null rate — Meta models lack music knowledge |
| qwen2.5:72b | 0.8 TPS on CPU — GPU only; not tested on GPU |
| deepseek-r1:32b | `think=False` ignored by ollama; 190s TTFT, 60s/song — unusable |
| phi4:14b | Std dev 6.4 — worst spread of any model tested |
| mistral-small3.2:24b | 34% null rate — Mistral training data gaps |
| qwen3.6:35b-a3b-q8_0 | 6.6 TPS / 4.20s/song vs Q4's 7.5 TPS / 3.67s/song — 14% slower with no quality gain (32-song test) |
| qwen3.6:27b-q8_0 | 1.0 TPS / 27.49s/song — Q8 at 27B hits memory bandwidth wall; ~80h for full library (32-song test) |
| command-r:35b | 44% null rate — Cohere training data has major music gaps (missed Janis Joplin, Eels, Nena, Beck); 2.0 TPS / 12.16s/song — 3.5× slower than qwen3.6:35b |

---

## Background: Model Types

### Reasoning models vs standard models

**Standard (non-reasoning) models** read the prompt and immediately start generating a response. They're fast because every token they produce is part of the answer you asked for.

**Reasoning models** (e.g. deepseek-r1, qwen3 with `think=True`) first generate a hidden "thinking" scratchpad — sometimes thousands of tokens of internal deliberation — before writing the actual answer. This produces more careful, calibrated output on hard problems (maths, logic, coding), but for a task like music rating it's overkill and very slow. deepseek-r1:32b took 190 seconds just to start outputting on DDR5 — useless for batch processing 10,000 songs. All tests here use `think=False` to disable this where the model supports it.

### Dense models vs MoE (Mixture of Experts)

**Dense models** activate all their parameters on every token. A 35B dense model uses all 35B parameters per token generated. This gives consistently high quality but is slower and uses more memory.

**MoE (Mixture of Experts) models** have a large total parameter count but only activate a small fraction per token — routing each token to a few specialist "expert" sub-networks. For example, qwen3:30b-instruct has 30B total parameters but only ~3B active per token. This makes it much faster than a 30B dense model (12.9 TPS vs 7.9 TPS for qwen3.6:35b) but the smaller active parameter count can mean shallower knowledge. In practice qwen3:30b-instruct's MoE architecture explains why it's faster than qwen3.6:35b despite similar overall size, but has a higher null rate (11% vs 8%).

### Model families

| Family | Maker | Music knowledge | Notes |
|--------|-------|----------------|-------|
| Qwen 2.5 / 3 / 3.6 | Alibaba | Excellent | Best results in testing; broad training data |
| Gemma 4 | Google | Good | Decent knowledge; slightly generous ratings |
| Mistral | Mistral AI | Average | High null rate (34%) — gaps in music training data |
| Llama 3.x | Meta | Poor | 38% null rate; Meta models underperform on music knowledge |
| Phi 4 | Microsoft | Average | Very narrow rating range (std dev 6.4); not suitable |
| DeepSeek-R1 | DeepSeek | Good calibration | Reasoning model; think=False ignored in ollama — 190s TTFT, unusable for batch |

---

## System Prompt

The prompt used for all benchmarks:

```
You are a music expert rating songs for a personal music library.

Respond with ONLY this JSON — no prose, no extra fields:
{
  "ratings": [
    {"id": 617, "rating": 72, "confidence": "high"},
    {"id": 3137, "rating": null, "confidence": "low"}
  ]
}

Rules:
- One entry per song; copy the id exactly from the input
- Include every song; order does not matter
- Set rating to null only when confidence is low (you don't recognise the song)
- Be sparing with 90+; most songs don't deserve it

Rating anchors:
  95  ABBA — Dancing Queen / AC/DC — Hells Bells  (all-time classic)
  82  ABBA — Waterloo  (good)
  70  ABBA — Does Your Mother Know  (average filler)
  50  3OH!3 — My First Kiss  (poor)

Target distribution (bell curve centred ~65):
  90-100 ~5%   (top classics)
  75-89  ~20%  (excellent / very good)
  50-74  ~50%  (average — peak of the bell)
  30-49  ~20%  (below average)
  1-29   ~5%   (poor / forgettable)
```

Note: despite the target distribution in the prompt, all models tested skew their mean high (71–79 vs target 65). This is a known limitation — models are trained to be agreeable and tend to rate generously. Prompt tuning (e.g. adding stronger anchors for low ratings, or examples of 30–50 rated songs) may help bring the mean closer to 65.

---

## Machine Summary

| Machine | RAM | Memory | Notes |
|---------|-----|--------|-------|
| DESKTOP-3GDSG0D (WSL2) | ~32GB | DDR4 | Dave's dev machine |
| bayanat301 (Proxmox) | 64GB | DDR5 | ~2x WSL2 TPS on same model |
| phil-gpu | TBD | GPU | Not yet tested |

DDR5 vs DDR4: approximately 2x TPS on same model (CPU inference is memory-bandwidth-bound).

---

## Key Findings

**Batch size 4 is optimal** on both DDR4 and DDR5, confirmed by sweep on qwen3:30b-instruct and qwen3.6:35b. Larger batches (5+) slow down per-song throughput.

**Null rate is training data, not hardware.** Models with high null rates (llama3.3, mistral, phi4) don't improve with faster hardware — they simply don't know the songs.

**MoE vs Dense:** qwen3:30b-instruct is MoE (~3B active params per token), giving 12.9 TPS on CPU. qwen3.6:35b is likely dense (~35B active), giving 7.9 TPS — but with significantly better quality (8% vs 11% null, 11.8 vs 9.5 std dev).

**Thinking models need `think=False`:** qwen3 and gemma4 family respect this via ollama. deepseek-r1 does not — TTFT stays at 190s regardless.

**Mean ratings skew high** (~70–79 vs target 65) across all models. qwen3.6:35b (71.7) is closest to target. This is a prompt calibration issue, not a model issue.

**Temperature 0.1** with `format=` structured output produces consistent, parseable results. The old ~14 std dev seen with qwen2.5:32b was at temp 0.8 without structured output — not comparable.

---

## Next Steps: GPU Testing

DDR5 CPU testing is complete — all viable CPU models have been benchmarked. The next step is GPU rental to test models that don't fit in 64GB of CPU RAM, starting with `qwen3.5:122b-a10b` — the first model with significantly more total knowledge than anything tested so far.

Token math: batch size 4 → 2,633 requests for full library; ~600 input + ~120 output tokens per request = ~1.58M input, ~0.32M output tokens total. On an A100 80GB, expect 50–150 TPS — a 200-song benchmark would take under 2 minutes per model.

### Priority models to test

| Model | Size (Q4) | Why |
|-------|-----------|-----|
| `qwen3.5:122b-a10b` | ~70GB | **Best bet.** Newer Qwen generation; 122B total params (10B active) — far more embedded knowledge than anything tested; just fits an 80GB GPU at Q4 |
| `qwen2.5:72b` | ~45GB | Was 0.8 TPS on CPU; quality ceiling for the Qwen 2.5 family; safe fallback if 3.5 unavailable |
| `qwen3:32b` | ~20GB | Dense Qwen3 32B — Alibaba states it matches qwen2.5:72b on benchmarks; comfortable fit, could run Q8 |
| `qwen3.6:35b-a3b-q8_0` | ~35GB | Current CPU winner at higher precision; on GPU, memory bandwidth isn't the bottleneck so Q8 may outperform Q4 |
| `qwen3.6:35b` | ~22GB | Confirm CPU winner's speed on GPU as a baseline |
| `qwen3:235b-a22b` | ~142GB Q4 — needs 2×A100/H100 | Qwen3 flagship MoE (235B total, 22B active) — likely highest open-source music knowledge available |
| `qwen3.5:397b-a17b` | ~230GB+ — needs 3-4×A100 | Qwen3.5 flagship |

### Likely not worth trying

| Model | Why to skip |
|-------|------------|
| `qwen3:72b` | Does not exist — Qwen3 dense tops out at 32B |
| `gemma4:27b` | 23% null on gemma4:26b already tested; benchmark-confirmed music-domain weaknesses — size won't fix training data gaps |
| Any Llama 3.x | 38% null rate on llama3.3:70b; Meta models lack music knowledge regardless of size |
| DeepSeek-R1 any size | `think=False` not respected in ollama — TTFT stays 190s+ |
| Phi-4 any size | Std dev 6.4 — compressed rating range is fundamental to its training, not a size issue |
| Mistral small/medium | 34% null rate — Mistral training data gaps in music knowledge |
| Groq (any model) | Only carries qwen3:32b and qwen3.6:27b, both already eliminated on CPU for quality (quality is model-intrinsic, not hardware) — doesn't carry qwen3.5:122b or qwen3:235b either |

### Cloud GPU costs (RunPod, full library run)

| GPU | VRAM | $/hr | Models that fit | Full library cost (incl. setup) |
|-----|------|------|-----------------|---------------------------------|
| A40 | 48GB | $0.44 | qwen3.6:35b, qwen3:32b | **~$0.70** (~1.5 hrs total) |
| RTX 6000 Ada | 48GB | $0.77 | same | ~$0.85 |
| A100 PCIe | 80GB | $1.39 | + qwen2.5:72b, qwen3.5:122b-a10b | **~$2–4** (more benchmarking time) |
| H200 | 141GB | $4.39 | + qwen3:235b-a22b | **~$5–8** |

**Vast.ai** often runs 20–40% cheaper than RunPod for the same GPU (A100 from ~$1.10/hr) — good for non-urgent batch jobs.

### Recommendation

**Start with RunPod A100 + `qwen3.5:122b-a10b`** — the first genuinely untested model with meaningfully more total knowledge than the CPU winner.

| Goal | Option | Cost |
|------|--------|------|
| Current winner on GPU, full run | RunPod A40 | ~$0.70, ~1.5 hr |
| Best single-GPU quality (122B) | RunPod A100 PCIe | ~$2–4 |
| Flagship quality (235B) | RunPod H200 | ~$5–8 |
