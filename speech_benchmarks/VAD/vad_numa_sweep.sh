#!/usr/bin/env bash
set -euo pipefail

# NUMA-based VAD benchmark sweep launcher.
#
# This script owns core parsing + process pinning and launches one benchmark run
# per (core_count, backend) combination:
#   numactl -C <core_list> -m 0 python vad_bench.py ...
#
# Microbenchmark reference (31.25 ms chunks / RTS terminology):
# https://github.com/snakers4/silero-vad/wiki/Performance-Metrics#silero-vad-performance-metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BENCH="${SCRIPT_DIR}/vad_bench.py"

MODE="standard"                 # standard | micro
ENGINE="silero"                 # silero | webrtcvad
CORES="1,2,4,8"                 # comma-separated counts
BACKENDS="both"                 # both | pytorch | onnx
RESULTS_DIR="${SCRIPT_DIR}/results"
RESULTS_CSV="${RESULTS_DIR}/vad_sweep_results.csv"
PYTHON_BIN="python3"
NUMA_NODE="0"
MICRO_ITERATIONS="5000"
REPEATS="1"
THRESHOLD="0.5"
BATCH_SIZE=""
BATCH_SIZE_EXPLICIT="false"
MICRO_BATCH_SIZES_DEFAULT="1,2,4,8"
WARMUP="true"

HF_DATASET="hf-internal-testing/librispeech_asr_dummy"
HF_CONFIG="clean"
HF_SPLIT="validation"
HF_NUM_SAMPLES="20"
HF_CACHE_DIR=""

AUDIO_DIR=""
LABELS_CSV=""
SYNTHETIC="false"
SYNTHETIC_FILES="4"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra_args_for_python]

Core sweep + NUMA pinning launcher for vad_bench.py.

Options:
  --mode <standard|micro>         Benchmark mode (default: standard)
  --engine <silero|webrtcvad>     Engine (default: silero)
  --cores <list>                  Comma-separated core counts (default: 1,2,4,8)
  --backends <both|pytorch|onnx>  Backend sweep for Silero (default: both)
  --results-dir <dir>             Results directory
  --results-csv <path>            Aggregate CSV output
  --python <path>                 Python executable (default: python3)
  --numa-node <id>                NUMA memory node for -m (default: 0)

  --micro-iterations <n>          Micro mode iterations (default: 5000)
  --repeats <n>                   Standard mode repeats per file (default: 1)
  --batch-size <n>                Logical batch size per timed step.
                                  In micro mode, default sweep is 1,2,4,8 unless this is set.
  --threshold <f>                 Silero threshold (default: 0.5)
  --no-warmup                     Disable warmup

  --audio-dir <dir>               Use local wav directory
  --labels-csv <path>             Optional labels CSV for standard accuracy metrics
  --synthetic                     Use synthetic dataset source
  --synthetic-files <n>           Synthetic file count (default: 4)

  --hf-dataset <name>             HF dataset (default: librispeech_asr_dummy)
  --hf-config <name>              HF config/subset (default: clean)
  --hf-split <name>               HF split (default: validation)
  --hf-num-samples <n>            HF max samples (default: 20)
  --hf-cache-dir <dir>            HF cache directory

  -h, --help                      Show help

Examples:
  $(basename "$0") --cores 1,2,4,8 --mode standard
  $(basename "$0") --mode micro --cores 1,2,4 --micro-iterations 20000
  $(basename "$0") --backends onnx --cores 1,2,4 --results-csv ./results/onnx.csv
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --cores) CORES="$2"; shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --results-csv) RESULTS_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --numa-node) NUMA_NODE="$2"; shift 2 ;;

    --micro-iterations) MICRO_ITERATIONS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; BATCH_SIZE_EXPLICIT="true"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --no-warmup) WARMUP="false"; shift ;;

    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --labels-csv) LABELS_CSV="$2"; shift 2 ;;
    --synthetic) SYNTHETIC="true"; shift ;;
    --synthetic-files) SYNTHETIC_FILES="$2"; shift 2 ;;

    --hf-dataset) HF_DATASET="$2"; shift 2 ;;
    --hf-config) HF_CONFIG="$2"; shift 2 ;;
    --hf-split) HF_SPLIT="$2"; shift 2 ;;
    --hf-num-samples) HF_NUM_SAMPLES="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;

    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v numactl >/dev/null 2>&1; then
  echo "ERROR: numactl is not available. Install numactl and retry." >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

IFS=',' read -r -a CORE_COUNTS <<< "${CORES}"
if [[ ${#CORE_COUNTS[@]} -eq 0 ]]; then
  echo "ERROR: empty core list." >&2
  exit 1
fi

BACKEND_LIST=()
case "${BACKENDS}" in
  both) BACKEND_LIST=(pytorch onnx) ;;
  pytorch) BACKEND_LIST=(pytorch) ;;
  onnx) BACKEND_LIST=(onnx) ;;
  *)
    echo "ERROR: invalid --backends value '${BACKENDS}'. Use both|pytorch|onnx." >&2
    exit 1
    ;;
esac

if [[ "${ENGINE}" != "silero" && "${BACKENDS}" == "both" ]]; then
  BACKEND_LIST=(pytorch)
fi

BATCH_SIZE_LIST=()
if [[ "${MODE}" == "micro" ]]; then
  if [[ "${BATCH_SIZE_EXPLICIT}" == "true" ]]; then
    BATCH_SIZE_LIST=("${BATCH_SIZE}")
  else
    IFS=',' read -r -a BATCH_SIZE_LIST <<< "${MICRO_BATCH_SIZES_DEFAULT}"
  fi
else
  if [[ "${BATCH_SIZE_EXPLICIT}" == "true" ]]; then
    BATCH_SIZE_LIST=("${BATCH_SIZE}")
  else
    BATCH_SIZE_LIST=("1")
  fi
fi

if [[ ${#BATCH_SIZE_LIST[@]} -eq 0 ]]; then
  echo "ERROR: empty batch size list." >&2
  exit 1
fi

for raw_batch_size in "${BATCH_SIZE_LIST[@]}"; do
  batch_size="$(echo "${raw_batch_size}" | xargs)"
  if ! [[ "${batch_size}" =~ ^[0-9]+$ ]] || [[ "${batch_size}" -lt 1 ]]; then
    echo "ERROR: invalid batch size '${batch_size}'" >&2
    exit 1
  fi
done

echo "Running sweep"
echo "  mode=${MODE}"
echo "  engine=${ENGINE}"
echo "  cores=${CORES}"
echo "  backends=${BACKEND_LIST[*]}"
echo "  batch_sizes=${BATCH_SIZE_LIST[*]}"
echo "  results_csv=${RESULTS_CSV}"

run_idx=0
for raw_core_count in "${CORE_COUNTS[@]}"; do
  core_count="$(echo "${raw_core_count}" | xargs)"
  [[ -z "${core_count}" ]] && continue

  if ! [[ "${core_count}" =~ ^[0-9]+$ ]] || [[ "${core_count}" -lt 1 ]]; then
    echo "ERROR: invalid core count '${core_count}'" >&2
    exit 1
  fi

  end_core=$((core_count - 1))
  cpu_list="0-${end_core}"

  for backend in "${BACKEND_LIST[@]}"; do
    for raw_batch_size in "${BATCH_SIZE_LIST[@]}"; do
      batch_size="$(echo "${raw_batch_size}" | xargs)"
      run_idx=$((run_idx + 1))
      echo "[run ${run_idx}] cores=${core_count} cpus=${cpu_list} backend=${backend} mode=${MODE} batch_size=${batch_size} OMP_NUM_THREADS=${core_count}"

      cmd=(
        "${PYTHON_BIN}" "${PY_BENCH}"
        --mode "${MODE}"
        --engine "${ENGINE}"
        --num-threads "${core_count}"
        --threshold "${THRESHOLD}"
        --repeats "${REPEATS}"
        --batch-size "${batch_size}"
        --micro-iterations "${MICRO_ITERATIONS}"
        --hf-dataset "${HF_DATASET}"
        --hf-config "${HF_CONFIG}"
        --hf-split "${HF_SPLIT}"
        --hf-num-samples "${HF_NUM_SAMPLES}"
        --synthetic-files "${SYNTHETIC_FILES}"
        --results-dir "${RESULTS_DIR}"
        --results-csv "${RESULTS_CSV}"
      )

      if [[ -n "${HF_CACHE_DIR}" ]]; then
        cmd+=(--hf-cache-dir "${HF_CACHE_DIR}")
      fi
      if [[ -n "${AUDIO_DIR}" ]]; then
        cmd+=(--audio-dir "${AUDIO_DIR}")
      fi
      if [[ -n "${LABELS_CSV}" ]]; then
        cmd+=(--labels-csv "${LABELS_CSV}")
      fi
      if [[ "${SYNTHETIC}" == "true" ]]; then
        cmd+=(--synthetic)
      fi
      if [[ "${WARMUP}" == "false" ]]; then
        cmd+=(--no-warmup)
      fi
      if [[ "${backend}" == "onnx" && "${ENGINE}" == "silero" ]]; then
        cmd+=(--onnx)
      fi
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
      fi

      OMP_NUM_THREADS="${core_count}" \
      numactl -C "${cpu_list}" -m "${NUMA_NODE}" "${cmd[@]}"
    done
  done
done

echo
echo "Sweep complete. Aggregated CSV: ${RESULTS_CSV}"
