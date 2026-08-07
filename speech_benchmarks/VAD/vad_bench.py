#!/usr/bin/env python3
"""
vad_bench.py
================================================================================
Single-run VAD benchmark for Silero/WebRTC engines.

This script does not perform process pinning or multi-core sweeps internally.
Use an external launcher (for example numactl/taskset in a bash script) to:
1) choose core sets,
2) pin CPU affinity,
3) set thread environment variables,
4) run per-core sweeps.

Modes
-----
standard (default)
    Runs over all .wav files in the selected dataset directory and computes:
    - RTF = total_infer_time / total_audio_time
    - RTS = 1 / RTF (Silero wiki naming: real-time speed)
    - per-file latency mean / median / p90 / p95
    - optional logical batch sizing via --batch-size

micro
    Uses exactly one wav file (first lexicographic file), chunks it into fixed
    31.25 ms windows (500 samples at 16 kHz), runs for N iterations, and reports:
    - mean_chunk_latency_us
    - RTF and RTS over the micro iterations
    - optional logical batch sizing via --batch-size

Reference:
https://github.com/snakers4/silero-vad/wiki/Performance-Metrics#silero-vad-performance-metrics

Accuracy metrics (precision/recall/F1/accuracy/FPR) are computed in standard
mode only when --labels-csv is supplied.
================================================================================
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

SAMPLE_RATE = 16000
FRAME_RESOLUTION_S = 0.01
MICRO_CHUNK_SEC = 0.03125
MICRO_CHUNK_SAMPLES = int(SAMPLE_RATE * MICRO_CHUNK_SEC)  # 500


# =============================================================================
# Synthetic dataset generation (optional fallback/source)
# =============================================================================
def generate_synthetic_dataset(out_dir, num_files=4, seed=1234):
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "labels.csv")

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "start", "end"])

        for file_idx in range(num_files):
            duration_s = rng.uniform(8, 15)
            n_samples = int(duration_s * SAMPLE_RATE)
            audio = rng.normal(0, 0.01, n_samples).astype("float32")

            t_cursor = rng.uniform(0.2, 1.0)
            segments = []
            while t_cursor < duration_s - 0.5:
                seg_dur = rng.uniform(0.4, 2.2)
                seg_dur = min(seg_dur, duration_s - t_cursor - 0.1)
                if seg_dur <= 0.1:
                    break
                start_sample = int(t_cursor * SAMPLE_RATE)
                end_sample = int((t_cursor + seg_dur) * SAMPLE_RATE)

                freqs = rng.uniform(150, 600, size=3)
                seg_len = end_sample - start_sample
                tt = np.arange(seg_len) / SAMPLE_RATE
                tone = sum(np.sin(2 * np.pi * fr * tt) for fr in freqs) / len(freqs)
                envelope = np.hanning(seg_len)
                audio[start_sample:end_sample] += 0.35 * tone * envelope
                audio[start_sample:end_sample] += rng.normal(0, 0.02, seg_len)

                segments.append((t_cursor, t_cursor + seg_dur))
                t_cursor += seg_dur + rng.uniform(0.3, 1.5)

            audio = np.clip(audio, -1.0, 1.0)
            fname = f"synthetic_{file_idx:02d}.wav"
            sf.write(os.path.join(out_dir, fname), audio, SAMPLE_RATE, subtype="PCM_16")

            for (s, e) in segments:
                writer.writerow([fname, f"{s:.3f}", f"{e:.3f}"])

    return out_dir, labels_path


# =============================================================================
# Hugging Face dataset loading
# =============================================================================
def prepare_hf_dataset(dataset_name, config, split, out_dir, max_samples=None, cache_dir=None):
    import soundfile as sf

    os.makedirs(out_dir, exist_ok=True)
    marker_path = os.path.join(out_dir, ".prepared")
    marker_contents = f"{dataset_name}|{config}|{split}|{max_samples}"

    if os.path.isfile(marker_path):
        with open(marker_path) as f:
            if f.read().strip() == marker_contents:
                existing = sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".wav"))
                if existing:
                    log(f"Using cached copy of {dataset_name} ({len(existing)} files) in {out_dir}")
                    return out_dir

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required to pull default Hugging Face audio. Install with:\n"
            "    pip install datasets soundfile\n"
            "...or pass --audio-dir / --synthetic to skip Hugging Face."
        )

    split_expr = split if not max_samples else f"{split}[:{max_samples}]"
    log(f"Loading Hugging Face dataset {dataset_name} (config={config}, split={split_expr}) ...")

    load_kwargs = {"split": split_expr}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    config = config or None
    ds = load_dataset(dataset_name, config, **load_kwargs) if config else load_dataset(dataset_name, **load_kwargs)

    n_written = 0
    for i, sample in enumerate(ds):
        audio = sample.get("audio")
        if audio is None:
            continue
        array = audio["array"]
        sr = audio["sampling_rate"]
        fname = f"hf_{i:04d}.wav"
        sf.write(os.path.join(out_dir, fname), array, sr, subtype="PCM_16")
        n_written += 1

    if n_written == 0:
        raise RuntimeError(f"No audio samples found in {dataset_name} (config={config}, split={split_expr})")

    with open(marker_path, "w") as f:
        f.write(marker_contents)

    log(f"Wrote {n_written} wav file(s) from {dataset_name} to {out_dir}")
    return out_dir


# =============================================================================
# Labels and scoring helpers
# =============================================================================
def load_labels(labels_csv):
    labels = {}
    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            labels.setdefault(fname, []).append((float(row["start"]), float(row["end"])))
    return labels


def segments_to_frame_labels(segments, duration_s, resolution_s=FRAME_RESOLUTION_S):
    n_frames = max(1, int(math.ceil(duration_s / resolution_s)))
    frames = [False] * n_frames
    for (s, e) in segments:
        start_f = max(0, int(s / resolution_s))
        end_f = min(n_frames, int(math.ceil(e / resolution_s)))
        for i in range(start_f, end_f):
            frames[i] = True
    return frames


def update_confusion(conf, ref_frames, hyp_frames):
    n = min(len(ref_frames), len(hyp_frames))
    for i in range(n):
        r, h = ref_frames[i], hyp_frames[i]
        if r and h:
            conf["tp"] += 1
        elif (not r) and h:
            conf["fp"] += 1
        elif r and (not h):
            conf["fn"] += 1
        else:
            conf["tn"] += 1


def confusion_to_metrics(conf):
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
    accuracy = (tp + tn) / total if total > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    return {
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
    }


# =============================================================================
# Engines
# =============================================================================
class SileroEngine:
    name = "silero"

    def __init__(self, num_threads, use_onnx=False):
        import torch

        self.use_onnx = use_onnx
        self.backend = "onnx" if use_onnx else "pytorch"
        torch.set_num_threads(max(1, num_threads))

        from silero_vad import load_silero_vad, get_speech_timestamps

        self._get_speech_timestamps = get_speech_timestamps
        self.model = load_silero_vad(onnx=use_onnx)

        # Best-effort ONNX session thread control to align with wiki guidance.
        if use_onnx:
            self._set_onnx_session_threads(max(1, num_threads))

    def _set_onnx_session_threads(self, num_threads):
        session = None
        for attr in ("session", "ort_session", "_session"):
            cand = getattr(self.model, attr, None)
            if cand is not None:
                session = cand
                break

        if session is None:
            log(
                "WARNING: ONNX model session object not found for explicit "
                "intra/inter-op thread assignment; relying on environment settings."
            )
            return

        assigned = False
        for attr in ("intra_op_num_threads", "inter_op_num_threads"):
            if hasattr(session, attr):
                try:
                    setattr(session, attr, int(num_threads))
                    assigned = True
                except Exception:
                    pass

        if not assigned:
            log(
                "WARNING: ONNX session does not expose writable intra/inter-op "
                "thread attributes; relying on environment settings."
            )

    def speech_segments(self, audio_np, sample_rate, threshold):
        import torch

        wav = torch.from_numpy(audio_np)
        ts = self._get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=sample_rate,
            threshold=threshold,
            return_seconds=True,
        )
        return [(seg["start"], seg["end"]) for seg in ts]


class WebRTCEngine:
    name = "webrtcvad"

    def __init__(self, num_threads, aggressiveness=2, frame_ms=30):
        import webrtcvad

        self.backend = "n/a"
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.num_threads = num_threads

    def speech_segments(self, audio_np, sample_rate, threshold):
        import numpy as np

        pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype("int16").tobytes()
        frame_bytes = int(sample_rate * (self.frame_ms / 1000.0)) * 2
        segments = []
        in_speech = False
        seg_start = 0.0
        t = 0.0
        step_s = self.frame_ms / 1000.0
        for i in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
            frame = pcm16[i : i + frame_bytes]
            is_speech = self.vad.is_speech(frame, sample_rate)
            if is_speech and not in_speech:
                in_speech = True
                seg_start = t
            elif (not is_speech) and in_speech:
                in_speech = False
                segments.append((seg_start, t))
            t += step_s
        if in_speech:
            segments.append((seg_start, t))
        return segments


def build_engine(engine_name, num_threads, use_onnx):
    if engine_name == "silero":
        return SileroEngine(num_threads, use_onnx=use_onnx)
    if engine_name == "webrtcvad":
        return WebRTCEngine(num_threads)
    raise ValueError(f"Unknown engine: {engine_name}")


# =============================================================================
# Audio loading
# =============================================================================
def read_wav_mono16k(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        target_len = int(duration * SAMPLE_RATE)
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype("float32")
        sr = SAMPLE_RATE
    return audio, sr


# =============================================================================
# CSV helpers
# =============================================================================
CSV_FIELDS = [
    "timestamp",
    "mode",
    "engine",
    "backend",
    "num_threads",
    "batch_size",
    "num_files",
    "micro_iterations",
    "micro_chunk_sec",
    "total_audio_sec",
    "total_infer_sec",
    "rtf",
    "rts",
    "throughput_audio_sec_per_wallclock_sec",
    "mean_latency_ms",
    "median_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "mean_chunk_latency_us",
    "threshold",
    "accuracy_enabled",
    "precision",
    "recall",
    "tpr",
    "f1",
    "accuracy",
    "fpr",
    "status",
]


def init_csv(path):
    exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not exists:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k) for k in CSV_FIELDS})


# =============================================================================
# Dataset source resolution
# =============================================================================
def resolve_audio_source(args):
    audio_dir = args.audio_dir
    labels_csv = args.labels_csv

    if audio_dir is not None:
        return audio_dir, labels_csv

    if args.synthetic:
        synthetic_dir = os.path.join(args.results_dir, "_synthetic_audio")
        log(f"--synthetic given: generating a synthetic labeled dataset in {synthetic_dir}")
        audio_dir, labels_csv = generate_synthetic_dataset(synthetic_dir, num_files=args.synthetic_files)
        return audio_dir, labels_csv

    hf_dir = os.path.join(args.results_dir, "_hf_audio", args.hf_dataset.replace("/", "_"))
    try:
        audio_dir = prepare_hf_dataset(
            args.hf_dataset,
            args.hf_config,
            args.hf_split,
            hf_dir,
            max_samples=args.hf_num_samples,
            cache_dir=args.hf_cache_dir,
        )
        if not labels_csv:
            log(
                "Note: selected Hugging Face dataset has no speech/silence timing labels, "
                "so accuracy metrics are disabled unless --labels-csv is provided."
            )
    except Exception as e:
        synthetic_dir = os.path.join(args.results_dir, "_synthetic_audio")
        log(
            f"WARNING: could not prepare Hugging Face dataset ({e}). "
            f"Falling back to synthetic labeled dataset in {synthetic_dir}"
        )
        audio_dir, labels_csv = generate_synthetic_dataset(synthetic_dir, num_files=args.synthetic_files)

    return audio_dir, labels_csv


# =============================================================================
# Benchmark modes
# =============================================================================
def safe_rts_from_rtf(rtf):
    return (1.0 / rtf) if (rtf is not None and rtf > 0) else None


def run_standard(args, engine, audio_dir, labels_csv):
    wav_files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))
    if not wav_files:
        raise RuntimeError(f"no .wav files found in {audio_dir}")

    labels = load_labels(labels_csv) if labels_csv else None
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    accuracy_enabled = labels is not None

    total_audio_s = 0.0
    total_infer_s = 0.0
    per_file_latency_ms = []

    for idx, fname in enumerate(wav_files):
        path = os.path.join(audio_dir, fname)
        audio, sr = read_wav_mono16k(path)
        duration_s = len(audio) / sr

        if args.warmup and idx == 0:
            engine.speech_segments(audio[: min(len(audio), sr)], sr, args.threshold)

        run_times = []
        hyp_segments = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for _ in range(args.batch_size):
                hyp_segments = engine.speech_segments(audio, sr, args.threshold)
            t1 = time.perf_counter()
            # Normalize batch wall-time to per-item latency so RTF stays
            # comparable across different batch-size settings.
            run_times.append((t1 - t0) / args.batch_size)

        best_time = min(run_times)
        total_audio_s += duration_s
        total_infer_s += best_time
        per_file_latency_ms.append(best_time * 1000.0)

        if accuracy_enabled:
            ref_segments = labels.get(fname, [])
            ref_frames = segments_to_frame_labels(ref_segments, duration_s)
            hyp_frames = segments_to_frame_labels(hyp_segments, duration_s)
            update_confusion(conf, ref_frames, hyp_frames)

    lat = sorted(per_file_latency_ms)

    def pct(p):
        if not lat:
            return None
        idx = min(len(lat) - 1, int(round(p / 100.0 * (len(lat) - 1))))
        return lat[idx]

    rtf = (total_infer_s / total_audio_s) if total_audio_s > 0 else None
    rts = safe_rts_from_rtf(rtf)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "standard",
        "engine": args.engine,
        "backend": getattr(engine, "backend", "n/a"),
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "num_files": len(wav_files),
        "micro_iterations": None,
        "micro_chunk_sec": None,
        "total_audio_sec": round(total_audio_s, 6),
        "total_infer_sec": round(total_infer_s, 6),
        "rtf": round(rtf, 6) if rtf is not None else None,
        "rts": round(rts, 3) if rts is not None else None,
        "throughput_audio_sec_per_wallclock_sec": round(rts, 3) if rts is not None else None,
        "mean_latency_ms": round(sum(lat) / len(lat), 3) if lat else None,
        "median_latency_ms": pct(50),
        "p90_latency_ms": pct(90),
        "p95_latency_ms": pct(95),
        "mean_chunk_latency_us": None,
        "threshold": args.threshold,
        "accuracy_enabled": accuracy_enabled,
        "status": "SUCCESS",
    }

    if accuracy_enabled:
        result.update(confusion_to_metrics(conf))
    else:
        result.update(
            {
                "precision": None,
                "recall": None,
                "tpr": None,
                "f1": None,
                "accuracy": None,
                "fpr": None,
            }
        )

    return result


def run_micro(args, engine, audio_dir):
    wav_files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))
    if not wav_files:
        raise RuntimeError(f"no .wav files found in {audio_dir}")

    sample_file = wav_files[0]
    sample_path = os.path.join(audio_dir, sample_file)
    audio, sr = read_wav_mono16k(sample_path)

    n_chunks = len(audio) // MICRO_CHUNK_SAMPLES
    if n_chunks < 1:
        raise RuntimeError(
            f"selected sample too short for {MICRO_CHUNK_SEC * 1000:.2f} ms chunking: {sample_path}"
        )

    chunks = [
        audio[i * MICRO_CHUNK_SAMPLES : (i + 1) * MICRO_CHUNK_SAMPLES]
        for i in range(n_chunks)
    ]

    if args.warmup:
        engine.speech_segments(chunks[0], sr, args.threshold)

    infer_times = []
    chunk_cursor = 0
    for i in range(args.micro_iterations):
        t0 = time.perf_counter()
        for _ in range(args.batch_size):
            chunk = chunks[chunk_cursor % len(chunks)]
            chunk_cursor += 1
            engine.speech_segments(chunk, sr, args.threshold)
        t1 = time.perf_counter()
        infer_times.append(t1 - t0)

    total_infer_s = sum(infer_times)
    total_chunks = args.micro_iterations * args.batch_size
    total_audio_s = total_chunks * MICRO_CHUNK_SEC
    rtf = (total_infer_s / total_audio_s) if total_audio_s > 0 else None
    rts = safe_rts_from_rtf(rtf)
    mean_chunk_latency_us = (total_infer_s / total_chunks) * 1e6 if total_chunks > 0 else None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "micro",
        "engine": args.engine,
        "backend": getattr(engine, "backend", "n/a"),
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "num_files": 1,
        "micro_iterations": args.micro_iterations,
        "micro_chunk_sec": MICRO_CHUNK_SEC,
        "total_audio_sec": round(total_audio_s, 6),
        "total_infer_sec": round(total_infer_s, 6),
        "rtf": round(rtf, 6) if rtf is not None else None,
        "rts": round(rts, 3) if rts is not None else None,
        "throughput_audio_sec_per_wallclock_sec": round(rts, 3) if rts is not None else None,
        "mean_latency_ms": None,
        "median_latency_ms": None,
        "p90_latency_ms": None,
        "p95_latency_ms": None,
        "mean_chunk_latency_us": round(mean_chunk_latency_us, 3) if mean_chunk_latency_us is not None else None,
        "threshold": args.threshold,
        "accuracy_enabled": False,
        "precision": None,
        "recall": None,
        "tpr": None,
        "f1": None,
        "accuracy": None,
        "fpr": None,
        "status": "SUCCESS",
    }


# =============================================================================
# CLI / main
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Single-run VAD benchmark. Use an external sweep launcher for "
            "core parsing, pinning, and repeated runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # standard benchmark over dataset using 4 threads
    ./vad_bench.py --mode standard --num-threads 4

    # microbenchmark with 31.25 ms chunks
    ./vad_bench.py --mode micro --num-threads 1 --micro-iterations 5000

    # ONNX backend
    ./vad_bench.py --mode standard --engine silero --onnx --num-threads 8
""",
    )

    p.add_argument("--mode", choices=["standard", "micro"], default="standard")
    p.add_argument("--engine", choices=["silero", "webrtcvad"], default="silero")
    p.add_argument("--onnx", action="store_true", help="Use ONNX backend for Silero (ignored for webrtcvad).")
    p.add_argument("--num-threads", type=int, default=1, help="Thread count for this single benchmark run.")
    p.add_argument("--batch-size", type=int, default=1, help="Logical batch size per timed step (default 1).")

    p.add_argument("--audio-dir", default=None, help="Directory of .wav files. If omitted, dataset is prepared from Hugging Face unless --synthetic is set.")
    p.add_argument("--labels-csv", default=None, help="Optional labels CSV (filename,start,end) for accuracy metrics in standard mode.")

    p.add_argument("--hf-dataset", default="hf-internal-testing/librispeech_asr_dummy", help="Hugging Face dataset used when --audio-dir is omitted.")
    p.add_argument("--hf-config", default="clean", help="Hugging Face dataset config/subset name (pass '' if none).")
    p.add_argument("--hf-split", default="validation", help="Hugging Face split.")
    p.add_argument("--hf-num-samples", type=int, default=20, help="Max number of clips to materialize from HF dataset.")
    p.add_argument("--hf-cache-dir", default=None, help="Optional datasets.load_dataset cache_dir.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic labeled audio as source.")

    p.add_argument("--threshold", type=float, default=0.5, help="Speech probability threshold (silero only).")
    p.add_argument("--repeats", type=int, default=1, help="Standard mode: repeats per file, best run kept.")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warmup pass.")
    p.add_argument("--micro-iterations", type=int, default=5000, help="Micro mode: number of 31.25 ms chunk inferences.")

    p.add_argument("--synthetic-files", type=int, default=4, help="Number of synthetic files when synthetic source is generated.")
    p.add_argument("--results-dir", default="./results", help="Directory for default generated assets.")
    p.add_argument("--results-csv", default=None, help="Optional CSV path to append this single run result.")

    return p


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.num_threads < 1:
        raise SystemExit("--num-threads must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.micro_iterations < 1:
        raise SystemExit("--micro-iterations must be >= 1")

    # Keep runtime threading aligned with user-requested thread count.
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    audio_dir, labels_csv = resolve_audio_source(args)

    use_onnx = bool(args.onnx and args.engine == "silero")
    engine = build_engine(args.engine, args.num_threads, use_onnx)

    try:
        if args.mode == "micro":
            result = run_micro(args, engine, audio_dir)
        else:
            result = run_standard(args, engine, audio_dir, labels_csv)
    except Exception as e:
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "engine": args.engine,
            "backend": "onnx" if use_onnx else ("pytorch" if args.engine == "silero" else "n/a"),
            "num_threads": args.num_threads,
            "batch_size": args.batch_size,
            "status": "FAILED",
            "error": str(e),
        }

    if args.results_csv:
        init_csv(args.results_csv)
        append_csv_row(args.results_csv, result)

    print(json.dumps(result))

    if result.get("status") != "SUCCESS":
        log(f"FAILED: {result.get('error', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
