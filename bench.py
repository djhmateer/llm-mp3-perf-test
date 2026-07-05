#!/usr/bin/env python3
# /// script
# dependencies = ["ollama", "pydantic"]
# ///
"""
Phase 9 — LLM benchmark.

For each model:
  1. Sweep batch sizes 1–20 to find the fastest songs-per-second throughput.
  2. Run the main benchmark at the optimal batch size and write results to CSV.

    uv run python phase9_perf_test/bench.py                         # all models, 100 songs (random sample from 10,531 — NOT all songs)
    uv run python phase9_perf_test/bench.py --songs 50 --models qwen2.5:32b
    uv run python phase9_perf_test/bench.py --machine phil-gpu

Fallback (no uv):
    pip install ollama pydantic
    python phase9_perf_test/bench.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import socket
import statistics
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timedelta
from pathlib import Path

import ollama
from pydantic import BaseModel
from typing import Literal, List, Optional

MODELS = ["qwen2.5:32b", "qwen2.5:72b", "qwen3:8b", "qwen3:14b", "qwen3:30b-instruct", "gemma4:12b", "gemma4:26b"]

DEFAULT_INPUT = Path(__file__).parent / "songs_export.jsonl"
DEFAULT_BASE_URL = "http://localhost:11434"

LOGS_DIR = Path(__file__).parent / "logs"

class _Rating(BaseModel):
    id: int
    rating: Optional[int] = None
    confidence: Literal['high', 'medium', 'low']

class _RatingsResponse(BaseModel):
    ratings: List[_Rating]

RATINGS_SCHEMA = _RatingsResponse.model_json_schema()

SWEEP_SEED = 1
BENCH_SEED = 42
SWEEP_REPEATS = 3    # runs per batch size during sweep
SWEEP_SONG_POOL = 40  # songs available to the sweep (largest batch = 20)
SWEEP_PLATEAU = 3    # stop after this many consecutive sizes worse than best

# ---------------------------------------------------------------------------
# Prompts  (mirrors phase5_enricher/enricher.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
"""


def build_user_prompt(songs: list[dict]) -> str:
    n = len(songs)
    lines = [f"Rate these {n} songs. Your response must contain exactly {n} ratings.\n"]
    for i, s in enumerate(songs, 1):
        artist = s.get("artist") or "Unknown"
        title = s.get("title") or "Unknown"
        lines.append(f"{i}. [ID:{s['id']}] Artist: {artist} | Title: {title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(
    client: ollama.Client,
    model: str,
    songs: list[dict],
    log: logging.Logger | None = None,
) -> tuple[str, float, float, int | None, int | None]:
    """Returns (content, ttft_s, total_s, completion_tokens, prompt_tokens)."""
    user_prompt = build_user_prompt(songs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if log:
        log.debug(f"\n=== USER MESSAGE ===\n{user_prompt.strip()}\n====================")

    is_thinking = bool(re.search(r'qwen3|gemma4|deepseek-r1', model, re.IGNORECASE))
    kwargs: dict = dict(
        model=model,
        messages=messages,
        stream=True,
        format=RATINGS_SCHEMA,
        options={"num_predict": len(songs) * 200, "temperature": 0.1},
    )
    if is_thinking:
        kwargs["think"] = False

    if log:
        log.debug(f"  call_model: model={model} songs={len(songs)} think={not is_thinking} num_predict={kwargs['options']['num_predict']} temperature={kwargs['options']['temperature']}")

    content = ""
    ttft_s: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    t_start = time.perf_counter()

    for chunk in client.chat(**kwargs):
        text = chunk.message.content or ""
        if text:
            if ttft_s is None:
                ttft_s = time.perf_counter() - t_start
            content += text
        if chunk.done:
            completion_tokens = chunk.eval_count
            prompt_tokens = chunk.prompt_eval_count
            if log:
                log.debug(f"  call_model done: prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} ttft={ttft_s:.3f}s total={time.perf_counter()-t_start:.3f}s")

    if log:
        log.debug(f"\n=== RAW RESPONSE ===\n{content.strip()}\n====================")

    return content, ttft_s or 0.0, time.perf_counter() - t_start, completion_tokens, prompt_tokens


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class SongResult:
    __slots__ = ("rating", "confidence", "parse_error", "id_matched")

    def __init__(self, rating: int | None, confidence: str | None, parse_error: bool, id_matched: bool = True):
        self.rating = rating
        self.confidence = confidence
        self.parse_error = parse_error
        self.id_matched = id_matched


def parse_ratings(text: str, songs: list[dict]) -> list[SongResult]:
    """Parse LLM response into per-song results, matching by echoed ID.

    Primary: match each response entry to a song by the 'id' field the model echoes back.
    Fallback: if an ID is missing or unrecognised, use the positional entry and flag it.

    parse_error=True  — JSON unparseable or structurally wrong (actual problem).
    rating=None with parse_error=False — model returned null intentionally (low confidence).
    id_matched=False  — ID was missing/wrong; result was assigned by position (log warning).
    """
    expected = len(songs)
    song_ids = [s["id"] for s in songs]

    def extract(entry: dict, id_matched: bool) -> SongResult:
        r = entry.get("rating")
        c = entry.get("confidence")
        rating = int(r) if isinstance(r, int) and 1 <= r <= 100 else None
        return SongResult(rating=rating, confidence=c, parse_error=False, id_matched=id_matched)

    parse_error_result = SongResult(None, None, parse_error=True, id_matched=False)

    try:
        data = json.loads(text)

        # single-song fallback
        if "rating" in data and expected == 1:
            return [extract(data, id_matched=True)]

        if "ratings" not in data:
            return [parse_error_result] * expected

        entries = [e for e in data["ratings"] if isinstance(e, dict)]

        # build id→entry map for ID-based lookup
        by_id: dict[int, dict] = {}
        for e in entries:
            eid = e.get("id")
            if isinstance(eid, int) and eid not in by_id:
                by_id[eid] = e

        out: list[SongResult] = []
        for pos, song_id in enumerate(song_ids):
            if song_id in by_id:
                out.append(extract(by_id[song_id], id_matched=True))
            elif pos < len(entries):
                # positional fallback — ID was missing or wrong
                out.append(extract(entries[pos], id_matched=False))
            else:
                out.append(parse_error_result)

        return out

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return [parse_error_result] * expected


# ---------------------------------------------------------------------------
# Key watcher — press N to skip to next model during sweep
# ---------------------------------------------------------------------------

def _watch_keys(skip_event: threading.Event) -> None:
    """Background thread: sets skip_event when N is pressed (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not skip_event.is_set():
            ch = sys.stdin.read(1)
            if ch.lower() == "n":
                skip_event.set()
                print("\n  [N pressed — skipping after current batch finishes...]")
                break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Batch-size sweep
# ---------------------------------------------------------------------------

def sweep_batch_sizes(client: ollama.Client, model: str, pool: list[dict], log: logging.Logger) -> int:
    """Try batch sizes 1–20, return the one with best median time-per-song."""
    print(f"\n  Part 1 — Batch-size sweep (fixed seed={SWEEP_SEED}, same {SWEEP_SONG_POOL} songs every run)")
    print(f"  Testing batch sizes 1–20, {SWEEP_REPEATS} runs each — press N to skip to next step")
    print(f"  {'Batch':>5}  {'median s/song':>13}  {'bar'}")
    print(f"  {'-'*5}  {'-'*13}  {'-'*20}")

    best_size = 1
    best_tps = float("inf")
    worse_streak = 0
    rng = random.Random(SWEEP_SEED)

    skip_event = threading.Event()
    watcher = threading.Thread(target=_watch_keys, args=(skip_event,), daemon=True)
    watcher.start()

    for batch_size in range(1, 21):
        if skip_event.is_set():
            print(f"  (skipped batch sizes {batch_size}–20)")
            break

        times_per_song: list[float] = []
        for run in range(SWEEP_REPEATS):
            if skip_event.is_set():
                break
            songs = rng.choices(pool, k=batch_size)
            try:
                _, _, total_s, _, _ = call_model(client, model, songs)  # no log — sweep is noisy
                times_per_song.append(total_s / batch_size)
                log.debug(f"  sweep batch={batch_size} run={run+1} total={total_s:.3f}s per_song={total_s/batch_size:.3f}s")
            except Exception as exc:
                log.error(f"  sweep batch={batch_size} run={run+1} ERROR: {exc}")

        if not times_per_song:
            print(f"  {batch_size:>5}  {'ERROR':>13}")
            log.error(f"  sweep batch={batch_size}: all {SWEEP_REPEATS} runs failed")
            continue

        median_tps = statistics.median(times_per_song)
        bar_len = min(20, int(20 * (1.0 / max(median_tps, 0.01)) / 10))
        bar = "█" * bar_len
        marker = " ◀ best" if median_tps < best_tps else ""
        print(f"  {batch_size:>5}  {median_tps:>12.3f}s  {bar}{marker}")
        log.info(f"  sweep batch={batch_size} median={median_tps:.3f}s/song{' [BEST]' if median_tps < best_tps else ''}")

        if median_tps < best_tps:
            best_tps = median_tps
            best_size = batch_size
            worse_streak = 0
        else:
            worse_streak += 1
            if worse_streak >= SWEEP_PLATEAU:
                print(f"  ({SWEEP_PLATEAU} consecutive sizes trending up — stopping sweep early)")
                log.info(f"  sweep early stop at batch_size={batch_size} after {SWEEP_PLATEAU} consecutive worse results")
                break

    skip_event.set()  # stop watcher if sweep finished naturally

    print(f"\n  Optimal batch size: {best_size} ({best_tps:.3f}s/song)")
    log.info(f"  sweep complete — optimal batch_size={best_size} ({best_tps:.3f}s/song)")
    return best_size


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def sanitize(model: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')


def plog(msg: str, log: logging.Logger) -> None:
    """Print to console and mirror to log file."""
    print(msg)
    log.info(msg)


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


RESULT_FIELDS = [
    "model", "song_id", "artist", "title", "batch_size",
    "ttft_s", "total_s", "tokens_generated", "tokens_per_sec", "prompt_tokens",
    "rating", "confidence", "parse_error", "id_matched", "cold_start", "raw_response",
]


def benchmark_model(
    client: ollama.Client,
    model: str,
    songs: list[dict],
    batch_size: int,
    log: logging.Logger,
) -> tuple[dict, list[dict]]:
    plog(f"\n  Part 2 — Running benchmark: {len(songs)} songs at batch_size={batch_size} on model {model}", log)

    rows: list[dict] = []
    offset = 0
    cold_start = True
    warm_elapsed_s = 0.0
    warm_songs = 0

    while offset < len(songs):
        batch = songs[offset:offset + batch_size]
        offset += len(batch)
        batch_label = f"songs {offset-len(batch)+1}–{offset}"

        try:
            content, ttft_s, total_s, tok, prompt_tok = call_model(client, model, batch, log)
        except Exception as exc:
            plog(f"    ERROR ({batch_label}): {exc}", log)
            log.error(f"  {batch_label} LLM call failed: {exc}")
            cold_start = False
            continue

        cold_tag = " [COLD START — excluded from averages]" if cold_start else ""
        log.info(f"  {batch_label} ttft={ttft_s:.3f}s total={total_s:.3f}s completion_tokens={tok or '?'} prompt_tokens={prompt_tok or '?'}{cold_tag}")

        results = parse_ratings(content, batch)
        n_rated = sum(1 for r in results if r.rating is not None)
        n_parse_errors = sum(1 for r in results if r.parse_error)
        if n_parse_errors:
            log.warning(f"  {batch_label} {n_parse_errors} parse error(s) — raw: {content.strip()}")
        else:
            log.debug(f"  {batch_label} parsed ok: {n_rated}/{len(batch)} rated (rest low-confidence null)")

        tps = tok / total_s if (tok and total_s > 0) else None
        time_per_song = total_s / len(batch)

        for i, (song, result) in enumerate(zip(batch, results)):
            artist = song.get("artist") or "Unknown"
            title = song.get("title") or "Unknown"
            if result.parse_error:
                log.warning(f"    song {song['id']} {artist} - {title}: PARSE ERROR")
            elif not result.id_matched:
                log.warning(f"    song {song['id']} {artist} - {title}: positional fallback (model returned wrong/missing id)")
            elif result.rating is not None:
                log.info(f"    song {song['id']} {artist} - {title}: rating={result.rating} confidence={result.confidence}")
            else:
                log.info(f"    song {song['id']} {artist} - {title}: null (confidence={result.confidence} - model does not recognise song)")
            rows.append({
                "model": model,
                "song_id": song["id"],
                "artist": artist,
                "title": title,
                "batch_size": batch_size,
                "ttft_s": f"{ttft_s:.3f}" if i == 0 else "",
                "total_s": f"{time_per_song:.3f}",
                "tokens_generated": tok or "",
                "tokens_per_sec": f"{tps:.1f}" if tps else "",
                "prompt_tokens": prompt_tok or "",
                "rating": result.rating or "",
                "confidence": result.confidence or "",
                "parse_error": result.parse_error,
                "id_matched": result.id_matched,
                "cold_start": cold_start,
                "raw_response": content.strip() if i == 0 else "",
            })

        status = f"{n_rated}/{len(batch)} rated"
        if n_parse_errors:
            status += f"  {n_parse_errors} parse-err"
        if cold_start:
            status += "  [cold start]"
        tps_str = f"{tps:.1f}" if tps else "?"

        if not cold_start:
            warm_elapsed_s += total_s
            warm_songs += len(batch)

        eta_str = ""
        remaining = len(songs) - offset
        if warm_songs and remaining > 0:
            avg_s_per_song = warm_elapsed_s / warm_songs
            eta = datetime.now() + timedelta(seconds=avg_s_per_song * remaining)
            eta_str = f"  eta={eta.strftime('%H:%M:%S')}"

        plog(f"    {batch_label:<12}  {status}  {time_per_song:.2f}s/song  tps={tps_str}{eta_str}", log)
        cold_start = False

    return _summarise(model, batch_size, rows, log), rows


def _summarise(model: str, batch_size: int, rows: list[dict], log: logging.Logger) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    warm_rows = [r for r in rows if not r.get("cold_start")]

    def avg(key: str, subset: list = None) -> float | None:
        src = subset if subset is not None else rows
        vals = [float(r[key]) for r in src if r.get(key) not in ("", None)]
        return sum(vals) / len(vals) if vals else None

    valid_pct = 100 * sum(1 for r in rows if not r.get("parse_error")) / n
    a_ttft = avg("ttft_s", warm_rows)
    a_total = avg("total_s", warm_rows)
    a_tps = avg("tokens_per_sec", warm_rows)

    # Rating quality stats
    rated_vals = [int(r["rating"]) for r in rows if r.get("rating") not in ("", None)]
    null_pct = 100 * sum(1 for r in rows if r.get("rating") in ("", None) and not r.get("parse_error")) / n
    pct_90plus = 100 * sum(1 for v in rated_vals if v >= 90) / len(rated_vals) if rated_vals else None
    rating_mean = sum(rated_vals) / len(rated_vals) if rated_vals else None
    rating_std = statistics.stdev(rated_vals) if len(rated_vals) >= 2 else None

    summary_lines = [
        f"\n  --- {model} summary ---",
        f"  Songs         : {n} ({n - len(warm_rows)} cold start excluded from timing averages)",
        f"  Optimal batch : {batch_size}",
        f"  Valid ratings : {valid_pct:.0f}%",
        f"  Null rate     : {null_pct:.0f}%  (model doesn't recognise song)",
        *(([f"  Rating mean   : {rating_mean:.1f}  (target ~65)"] if rating_mean else [])),
        *(([f"  Rating std dev: {rating_std:.1f}  (higher = better spread)"] if rating_std else [])),
        *(([f"  Rated 90+     : {pct_90plus:.0f}%  (target ~5%)"] if pct_90plus is not None else [])),
        *(([f"  Avg TTFT      : {a_ttft:.2f}s"] if a_ttft else [])),
        *(([f"  Avg s/song    : {a_total:.2f}s"] if a_total else [])),
        *(([f"  Avg tokens/s  : {a_tps:.1f}"] if a_tps else [])),
    ]
    for line in summary_lines:
        print(line)
        log.info(line.strip())

    return {
        "model": model, "batch_size": batch_size, "n": n,
        "valid_pct": valid_pct, "null_pct": null_pct,
        "rating_mean": rating_mean, "rating_std": rating_std, "pct_90plus": pct_90plus,
        "avg_ttft_s": a_ttft, "avg_total_s": a_total, "avg_tps": a_tps,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_songs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM inference across models")
    parser.add_argument("--songs", type=int, default=100, help="Songs to benchmark (default 100)")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model list")
    parser.add_argument("--machine", default=socket.gethostname(), help="Label for output files")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Songs JSONL file")
    parser.add_argument("--force-batch-size", type=int, default=None, metavar="N",
                        help="Skip Part 1 sweep and run Part 2 directly at this batch size")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run export_songs.py first.", file=sys.stderr)
        sys.exit(1)

    LOGS_DIR.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"bench_{run_ts}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
    log = logging.getLogger("bench")
    out_path = Path(__file__).parent / f"results_{args.machine}_{run_ts}.csv"
    t_total_start = time.perf_counter()
    log.info(f"bench.py start — machine={args.machine} models={args.models} songs={args.songs}")
    log.debug(f"\n=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT.strip()}\n====================\n")
    print(f"Log      : {log_file}")
    print(f"Results  : {out_path}")

    all_songs = load_songs(args.input)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = ollama.Client(host=args.base_url)

    # Fixed sample for the main benchmark (reproducible across models)
    rng = random.Random(BENCH_SEED)
    bench_songs = rng.sample(all_songs, min(args.songs, len(all_songs)))

    # Fixed pool for the sweep (disjoint seed from bench sample)
    sweep_pool = random.Random(SWEEP_SEED).sample(all_songs, min(SWEEP_SONG_POOL, len(all_songs)))

    if args.songs < 20:
        print(f"WARNING: --songs {args.songs} is less than 20; the main benchmark won't exercise all batch sizes.", file=sys.stderr)

    print(f"Machine        : {args.machine}")
    print(f"Songs (Part 2) : {len(bench_songs)}")
    print(f"Models         : {models}")
    print(f"Ollama         : {args.base_url}")

    summaries: list[dict] = []
    all_rows: list[dict] = []
    model_times: list[tuple[str, float]] = []
    for model in models:
        plog(f"\n{'='*60}", log)
        plog(f"Model: {model}", log)
        plog(f"{'='*60}", log)

        t_model_start = time.perf_counter()
        if args.force_batch_size is not None:
            optimal = args.force_batch_size
            plog(f"  Skipping Part 1 — using forced batch size: {optimal}", log)
            log.info(f"  sweep skipped — forced batch_size={optimal}")
        else:
            optimal = sweep_batch_sizes(client, model, sweep_pool, log)

        summary, rows = benchmark_model(client, model, bench_songs, optimal, log)
        model_elapsed = time.perf_counter() - t_model_start
        summaries.append(summary)
        all_rows.extend(rows)
        model_times.append((model, model_elapsed))

    # Write single sorted CSV for the whole run
    sorted_rows = sorted(
        all_rows,
        key=lambda r: int(r["rating"]) if r.get("rating") not in ("", None) else -1,
        reverse=True,
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)
    log.info(f"Results written to {out_path} ({len(all_rows)} rows)")

    total_elapsed = time.perf_counter() - t_total_start

    plog(f"\n{'='*60}", log)
    plog("OVERALL SUMMARY", log)
    plog(f"{'='*60}", log)
    plog(f"{'Model':<20} {'Batch':>5} {'Valid%':>6} {'Null%':>6} {'Mean':>5} {'Std':>5} {'90+%':>5} {'TTFT':>7} {'s/song':>7} {'TPS':>5}", log)
    plog("-" * 75, log)
    for s in summaries:
        if not s:
            continue
        plog(
            f"{s['model']:<20}"
            f"  {s['batch_size']:>4}"
            f"  {s['valid_pct']:>5.0f}%"
            f"  {s.get('null_pct') or 0:>5.0f}%"
            f"  {s.get('rating_mean') or 0:>5.1f}"
            f"  {s.get('rating_std') or 0:>5.1f}"
            f"  {s.get('pct_90plus') or 0:>4.0f}%"
            f"  {s['avg_ttft_s'] or 0:>5.2f}s"
            f"  {s['avg_total_s'] or 0:>5.2f}s"
            f"  {s['avg_tps'] or 0:>4.1f}",
            log,
        )
    plog("", log)
    plog("Timing:", log)
    for model, elapsed in model_times:
        plog(f"  {model:<20}  {fmt_elapsed(elapsed)}", log)
    plog(f"  {'Total':<20}  {fmt_elapsed(total_elapsed)}", log)
    plog(f"\nResults : {out_path}", log)
    log.info(f"Run complete — {len(models)} model(s), {len(all_rows)} total rows")


if __name__ == "__main__":
    main()
