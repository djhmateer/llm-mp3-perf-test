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

## Full-library CPU runs (10,531 songs, reproducibility check)

Two independent full-library runs on the same CPU machine (`bayanat301`, DDR5), one day apart, same model/config as the 200-song CPU winner above.

| Metric | Run 1 (Jun 30) | Run 2 (Jul 1) | Target |
|--------|----------------|----------------|--------|
| TPS | 7.4 | 7.2 | — |
| s/song | 3.70s | 3.81s | — |
| Total wall time | 10.81h | 11.15h | — |
| Null rate | 4.1% | 4.1% | <20% |
| Mean | 71.3 | 71.3 | ~65 |
| Std dev | 12.1 | 12.1 | >10 |
| 90+% | 3.5% | 3.6% | ~5% |
| Valid ratings | 100% | 100% | 100% |

Aggregate numbers match the 200-song sample closely and are essentially identical between the two full runs — strong confirmation the earlier sample generalizes to the whole library.

**Per-song non-determinism, however, is real.** Comparing individual ratings between the two runs (same songs, same model, "fixed random seed, temp 0.1" per the stated methodology):

- **8,229 / 10,531 songs (78%) got the exact same rating** in both runs.
- **2,251 songs (21%) got a different rating** — median difference **6 points**, some swinging by up to **47 points**.
- 349 songs (3.3% of the library) differed by more than 10 points between runs.

Likely cause: Ollama's batched/continuous inference (context caching, batch composition timing, etc.) introduces small numeric differences that compound over an 11-hour run, even at low temperature with a fixed seed. **Aggregate statistics (mean, std dev, null rate) are solid and reproducible — but no single song's rating should be treated as precise.** Practical implication for using these ratings (e.g. playlist thresholds): treat any individual score as ±6 points of noise; borderline songs near a cutoff may land on either side depending on which run generated them.

---

## GPU: qwen3.6:35b on RunPod (RTX A5000, 24GB)

Same model and weights as the CPU winner above, run on a RunPod RTX A5000 pod (200-song run, batch size 4) to see how much a 24GB GPU speeds things up.

| Metric | GPU (RTX A5000) | CPU (DDR5, winner above) | Target |
|--------|-----------------|--------------------------|--------|
| TPS | **45.5** | 7.9 | — |
| s/song | **0.60s** | 3.48s | — |
| Std dev | 12.4 | 11.8 | >10 |
| Null rate | **7%** | 8% | <20% |
| 90+% | 4.8% | 5% | ~5% |
| Mean | 71.2 | 71.7 | ~65 |
| Valid ratings | 100% | 100% | 100% |
| Full library est. | **~1.76h** | ~10.1h | — |

**~5.7x faster** than CPU, with quality metrics essentially unchanged (same model — this is a sanity check that GPU inference doesn't degrade output, not a different comparison point). Null rate actually came in slightly better than CPU (7% vs 8%) — within normal run-to-run variance.

At $0.27/hr, the RTX A5000 turns qwen3.6:35b from "best quality but slow" into fast enough to just always use — full library (10,531 songs) in under 2 hours for a few cents.

---

## Tested and eliminated: qwen3.5:122b-a10b (A100 PCIe, 80GB)

Hypothesis going in: 122B total / 10B active params (vs qwen3.6:35b's 35B/3B) should mean meaningfully more embedded music knowledge. **Result: it's worse on every quality metric that matters, and much slower/costlier.** 200-song run, batch size 4.

| Metric | qwen3.5:122b-a10b (A100) | qwen3.6:35b (A5000, winner) | Target |
|--------|--------------------------|------------------------------|--------|
| TPS | 11.1 | 45.5 | — |
| s/song | 2.45s | 0.60s | — |
| Null rate | 20% | **7%** | <20% |
| Mean | 74.1 | 71.2 | ~65 |
| Std dev | 11.4 | 12.4 | >10 |
| 90+% | 8% | 4.8% | ~5% |
| Valid ratings | 100% | 100% | 100% |
| Full library est. | ~7.2h | ~1.76h | — |
| Full library cost | ~$10 (A100 @ $1.39/hr) | ~$0.47 (A5000 @ $0.27/hr) | — |

More active parameters didn't translate to better music knowledge here — null rate is *worse* (20% vs 7%, right at the edge of the acceptable threshold) and the mean drifts further from the target. Speed is ~4x slower (compute-bound on the larger active-parameter count, not a GPU/offload problem — confirmed all 49/50 layers resident on the A100), and a full-library run would cost ~20x more than the current winner. **Bigger is not automatically better for this task** — likely because whatever training-data/RLHF choices shape music recall don't scale simply with active params. Sticking with `qwen3.6:35b` as the winner.

---

## Tested and eliminated: qwen3:235b-a22b (B200, 142GB)

Qwen3's largest self-hostable flagship (235B total / 22B active). Smoke-tested at 32 songs before committing to a full 200-song run, given the cost (B200 @ $5.89/hr) and the qwen3.5:122b-a10b result above.

| Metric | Value |
|--------|-------|
| TPS | 12.3 |
| s/song | 1.52s |
| Reported null rate | 78% (25/32) |
| Mean (rated only) | 80.1 |
| Std dev | 6.4 |
| 90+% | 0% |

**Reported 78% null rate is misleading — this is a format bug, not a knowledge gap.** Every raw response is well-formed JSON, correctly matched to the right song IDs (`parse_error=False`, `id_matched=True` on every row) — this is not a batch-size-4 or parsing problem. Breaking down the 25 "null" rows by what the model actually said:

| Type | Count | Example |
|------|-------|---------|
| Genuine unknown song (`confidence: low`) | 5 | Blood Ceremony, a BBC Radio 4 comedy clip |
| **`confidence: high`, but the model omits the `rating` key entirely** | **20** | The Smiths — *There is a Light...*, Nena — *99 Luftballons*, Janis Joplin — *Piece Of My Heart*, Oasis, Beck, Metallica, Eels |

That second group are extremely well-known songs every other model tested (including the winner) rated confidently and correctly. `qwen3:235b-a22b` says `"confidence": "high"` for them — meaning it does recognize the song — then produces e.g. `{"id": 13701, "confidence": "high"}` with no `rating` field at all, instead of `{"id": 13701, "confidence": "high", "rating": 78}`. `bench.py`'s `entry.get("rating")` (bench.py:197) can't distinguish "intentionally null because low confidence" from "key just missing" — both count as null — which is why the metric looks catastrophic.

**True null rate (songs it actually doesn't recognize) is ~16% (5/32)** — within the <20% target. But this doesn't rescue the model: unreliable adherence to "always include a rating" makes it unfit for unattended batch processing regardless of underlying knowledge, similar in spirit to why `deepseek-r1` was eliminated (structurally plausible output that doesn't follow the format contract). **Not pursuing a full 200-song run or prompt-engineering fix** — not worth the B200 cost to rescue a formatting failure on the flagship model when the current winner has no such issue.

---

## B200 speed check: qwen3.6:35b (same pod as qwen3:235b-a22b test)

After eliminating `qwen3:235b-a22b`, re-ran the winner model on the same B200 pod (200-song run) as a quick sanity/speed check — not a new model result, since it's the same weights already benchmarked on the A5000, but a useful data point on hardware tradeoffs.

| Metric | B200 | RTX A5000 (recommended) |
|--------|------|--------------------------|
| TPS | **125.3** | 45.5 |
| s/song | **0.22s** | 0.60s |
| Null rate | 8% | 7% |
| Mean | 71.0 | 71.2 |
| Std dev | 11.6 | 12.4 |
| 90+% | 3.8% | 4.8% |
| Full library est. | **~38 min** | ~1.76h |
| Full library cost | ~$3.76 (B200 @ $5.89/hr) | ~$0.47 (A5000 @ $0.27/hr) |

Quality is identical within noise (as expected — same model, same weights), confirming output doesn't depend on which GPU runs it. B200 is ~2.75x faster than the A5000 but costs ~8x more for the full library run. **Stick with the A5000 for routine full-library runs** — only reach for the B200 if wall-clock time matters more than the ~$3.30 cost difference (e.g. want the full library done in under 40 minutes rather than ~1h45).

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

DDR5 CPU testing is complete. GPU testing has confirmed `qwen3.6:35b` as a fast, cheap winner (A5000) and eliminated both large flagship attempts: `qwen3.5:122b-a10b` (A100, genuine recall gap) and `qwen3:235b-a22b` (B200, format-adherence bug — see above). Bigger active-parameter count has not translated to better music knowledge or reliability in either case, just more runtime and cost. That's worth keeping in mind for what to try next: prioritize models likely to have *better training data for music*, not just more parameters.

Also confirmed since the original research: `qwen3.5:397b` (the true Qwen 3.5 flagship) is **cloud-only** via Ollama (`qwen3.5:397b-cloud`) — it can't be pulled and self-hosted on a rented GPU, so it's off the table regardless of budget.

Token math: batch size 4 → 2,633 requests for full library; ~600 input + ~120 output tokens per request = ~1.58M input, ~0.32M output tokens total.

### Priority models to test

| Model | Size (Q4) | Why |
|-------|-----------|-----|
| `qwen2.5:72b` | ~45GB | **Best bet next.** Ran on CPU (0.8 TPS) but was never quality-tested — too slow for a real sample. Now cheap to test properly on GPU (fits A40 $0.44/hr or A100). Different generation/training data than qwen3.6, so may recall music qwen3.6 doesn't |
| `qwen3:32b` | ~20GB | Dense Qwen3 32B — Alibaba states it matches qwen2.5:72b on benchmarks; comfortable fit on any GPU tested so far, could run Q8 for extra precision at this size |
| `qwen3.6:35b-a3b-q8_0` | ~35GB | Current winner at higher precision; on GPU, memory bandwidth isn't the bottleneck so Q8 may outperform Q4 (was 14% slower with no quality gain on CPU — GPU could flip this) |

### Likely not worth trying

| Model | Why to skip |
|-------|------------|
| `qwen3.5:122b-a10b` | **Tested and eliminated** — 20% null rate (vs 7% for winner), 4x slower, ~20x costlier full-library run; see comparison above |
| `qwen3:235b-a22b` | **Tested and eliminated** — reported 78% null rate on a 32-song smoke test, though ~16/32 of that is actually a format bug (model says `confidence: high` but omits `rating` for well-known songs) rather than a true knowledge gap; unreliable format adherence disqualifies it regardless — see above |
| `qwen3.5:397b` | Cloud-only (`-cloud` tag) — not self-hostable on a rented GPU |
| `qwen3:72b` | Does not exist — Qwen3 dense tops out at 32B |
| `gemma4:27b` | 23% null on gemma4:26b already tested; benchmark-confirmed music-domain weaknesses — size won't fix training data gaps |
| Any Llama 3.x | 38% null rate on llama3.3:70b; Meta models lack music knowledge regardless of size |
| DeepSeek-R1 any size | `think=False` not respected in ollama — TTFT stays 190s+ |
| Phi-4 any size | Std dev 6.4 — compressed rating range is fundamental to its training, not a size issue |
| Mistral small/medium | 34% null rate — Mistral training data gaps in music knowledge |
| Groq (any model) | Only carries qwen3:32b and qwen3.6:27b, both already eliminated on CPU for quality (quality is model-intrinsic, not hardware) — doesn't carry qwen2.5:72b or qwen3:235b either |

### Cloud GPU costs (RunPod, full library run)

| GPU | VRAM | $/hr | Models that fit | Full library cost (incl. setup) |
|-----|------|------|-----------------|---------------------------------|
| RTX A5000 | 24GB | $0.27 | qwen3.6:35b (**winner, confirmed**) | **~$0.47** (~1.75 hrs total) |
| A40 | 48GB | $0.44 | + qwen2.5:72b, qwen3:32b | ~$0.70–1 |
| RTX 6000 Ada | 48GB | $0.77 | same | ~$0.85–1.20 |
| A100 PCIe | 80GB | $1.39 | qwen3.5:122b-a10b (**eliminated**) | ~$10, confirmed |
| B200 | 180GB | $5.89 | qwen3:235b-a22b (**eliminated**) | ~$3.76 confirmed (qwen3.6:35b speed check, see above) |

**Vast.ai** often runs 20–40% cheaper than RunPod for the same GPU (A100 from ~$1.10/hr) — good for non-urgent batch jobs.

### Recommendation

**Next: RunPod A40 + `qwen2.5:72b`** — a different model generation/training set than qwen3.6, never quality-tested (too slow on CPU to get a real sample), and cheap to test now that GPU speed makes 200-song runs take minutes. Given both large-flagship results above, don't assume it'll beat the winner — but it's the cheapest remaining unknown. Both flagship scaling bets (122B and 235B) are now closed out; no further "bigger model" tests are planned.

| Goal | Option | Cost |
|------|--------|------|
| Current winner, full run | RunPod A5000 | ~$0.47, ~1.75 hr |
| Next test: qwen2.5:72b | RunPod A40 | ~$0.10–0.20 for a 200-song sample |

RunPod/Vast.ai
