# VAD Benchmarking

This directory provides a clean two-script benchmarking workflow.

- Single-run benchmark script: vad_bench.py
- NUMA-pinned sweep launcher: vad_numa_sweep.sh

The Python script does not pin cores or run internal sweeps. Core parsing, pinning, and sweeps are handled by the bash launcher.

Silero microbenchmark reference:
https://github.com/snakers4/silero-vad/wiki/Performance-Metrics#silero-vad-performance-metrics

## What You Get

### Standard mode

- Dataset-wide benchmarking across wav files
- RTF and RTS
- Latency stats: mean, median, p90, p95
- Optional frame-level accuracy metrics if labels are provided

### Micro mode

- Uses one audio file from the prepared HF/local source
- Chunks into 31.25 ms windows at 16 kHz
- Reports mean chunk latency in microseconds, plus RTF and RTS

### Batch size

- New option: batch size
- Exposed as --batch-size in both scripts
- Standard mode default: 1
- Micro mode default sweep: 1,2,4,8
- If --batch-size is explicitly provided, only that size is used
- Applied as logical timed batch size in both standard and micro modes

## Requirements

Install Python dependencies:

pip install -r requirements.txt

Install system dependency for NUMA pinning:

- Ubuntu or Debian:
  sudo apt-get install -y numactl

## Default Sweep Behavior

By default the sweep launcher runs:

- Core counts: 1,2,4,8
- Backends: both (pytorch and onnx for Silero)
- Mode: standard

Pinned command style per run:

numactl -C core_list -m 0 python ...

Thread environment set per run:

- OMP_NUM_THREADS=core_count

## Quick Start

Run from this directory:

cd /nfs_home/akulshre/scripts/speech_benchmarks/VAD

Default standard sweep:

./vad_numa_sweep.sh

Standard sweep with explicit cores:

./vad_numa_sweep.sh --cores 1,2,4,8

Micro sweep:

./vad_numa_sweep.sh --mode micro --cores 1,2,4,8 --micro-iterations 20000

Micro sweep with explicit single batch size override:

./vad_numa_sweep.sh --mode micro --cores 1,2,4,8 --batch-size 8 --micro-iterations 20000

## Common Sweep Options

Mode:

- --mode standard
- --mode micro

Backends:

- --backends both
- --backends pytorch
- --backends onnx

Engine:

- --engine silero
- --engine webrtcvad

Performance knobs:

- --repeats N
- --batch-size N (forces a single size; otherwise micro mode sweeps 1,2,4,8)
- --micro-iterations N
- --threshold F
- --no-warmup

Data source:

- --audio-dir /path/to/wavs
- --labels-csv /path/to/labels.csv
- --synthetic
- --hf-dataset DATASET
- --hf-config CONFIG
- --hf-split SPLIT
- --hf-num-samples N
- --hf-cache-dir /path/to/cache

Output control:

- --results-dir DIR
- --results-csv FILE

## Local Audio with Labels

Example for standard mode accuracy metrics:

./vad_numa_sweep.sh \
  --mode standard \
  --audio-dir /path/to/wavs \
  --labels-csv /path/to/labels.csv \
  --cores 1,2,4,8

labels.csv format:

filename,start,end
sample1.wav,0.42,1.10
sample1.wav,2.00,3.35
sample2.wav,0.00,0.80

## Single Run Without Sweep

Standard:

python3 vad_bench.py --mode standard --num-threads 4 --batch-size 1

Micro:

python3 vad_bench.py --mode micro --num-threads 4 --batch-size 8 --micro-iterations 20000

ONNX backend:

python3 vad_bench.py --mode standard --engine silero --onnx --num-threads 4 --batch-size 1

## Output Metrics

Common:

- rtf
- rts
- throughput_audio_sec_per_wallclock_sec

Standard-only:

- mean_latency_ms
- median_latency_ms
- p90_latency_ms
- p95_latency_ms
- precision, recall, f1, accuracy, fpr, tpr when labels are present

Micro-only:

- mean_chunk_latency_us

Recorded metadata:

- mode
- engine
- backend
- num_threads
- batch_size

## Result CSV

Default aggregate CSV from sweep launcher:

./results/vad_sweep_results.csv

Override path:

./vad_numa_sweep.sh --results-csv /tmp/vad_results.csv
