#!/usr/bin/env bash
#
# run_benchmark.sh - convenience wrapper around the oneCCL "benchmark" tool
# (installed with oneAPI at $CCL_ROOT/share/doc/ccl/examples/benchmark) to
# benchmark collective communication operations (allreduce, allgather,
# alltoall, bcast, reduce, reduce_scatter, ...) on CPU or Intel GPU (XPU),
# for a given tensor size(s) and number of devices/ranks.
#
# Results are written to a CSV file and also printed to the terminal.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/build/benchmark/benchmark"

ALL_OPS="allgather,allgatherv,allreduce,alltoall,alltoallv,bcast,broadcast,reduce,reduce_scatter"
ALL_DTYPES="int8,int32,int64,uint64,float16,float32,float64,bfloat16"

# ---- defaults ----------------------------------------------------------
BACKEND="cpu"            # cpu | xpu
OPS="allreduce"           # comma list, or "all"
SIZES="1024,262144,16777216"   # element counts (comma list) -> -y
DEVICES=1                 # number of ranks/devices -> mpirun -n
PPN=""                    # processes per node, default = DEVICES
HOSTFILE=""
DTYPE="float32"
ITERS=16
WARMUP_ITERS=16
OUTDIR="${SCRIPT_DIR}/results"
TAG=""
EXTRA_ARGS=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmarks oneCCL collective operations on CPU or Intel GPU (XPU).

Options:
  --backend <cpu|xpu>       Compute backend (default: ${BACKEND})
  --ops <list|all>          Comma-separated collectives to run (default: ${OPS})
                             available: ${ALL_OPS}
  --sizes <list>            Comma-separated element counts per buffer (default: ${SIZES})
  --devices <N>             Number of ranks/devices, i.e. mpirun -n (default: ${DEVICES})
  --ppn <N>                 Ranks per node, i.e. mpirun -ppn (default: = --devices)
  --hostfile <path>         Optional MPI hostfile for multi-node runs
  --dtype <list|all>        Comma-separated datatypes (default: ${DTYPE})
                             available: ${ALL_DTYPES}
  --iters <N>               Measured iterations per size (default: ${ITERS})
  --warmup-iters <N>        Warm-up iterations per size (default: ${WARMUP_ITERS})
  --outdir <path>            Directory for CSV/log output (default: ${OUTDIR})
  --tag <name>               Extra label included in the output file names
  --extra "<args>"           Extra raw arguments passed through to the benchmark binary
  -h, --help                  Show this help and exit

Examples:
  # CPU, single node, 4 ranks, allreduce + allgather over 3 sizes
  $(basename "$0") --backend cpu --devices 4 --ops allreduce,allgather --sizes 4096,1048576

  # XPU (GPU), 2 ranks (2 tiles/GPUs on one node)
  $(basename "$0") --backend xpu --devices 2 --ops allreduce --sizes 1048576,16777216
EOF
}

# ---- arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --ops) OPS="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --ppn) PPN="$2"; shift 2 ;;
        --hostfile) HOSTFILE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --extra) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${BACKEND}" != "cpu" && "${BACKEND}" != "xpu" ]]; then
    echo "ERROR: --backend must be 'cpu' or 'xpu' (got '${BACKEND}')" >&2
    exit 1
fi

[[ -z "${PPN}" ]] && PPN="${DEVICES}"

# ---- environment checks --------------------------------------------------
if [[ -z "${CCL_ROOT:-}" || -z "${I_MPI_ROOT:-}" ]]; then
    echo "ERROR: oneAPI environment not detected (CCL_ROOT/I_MPI_ROOT unset)." >&2
    echo "  source /swtools/intel/oneapi/setvars.sh" >&2
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found on PATH after sourcing the oneAPI environment." >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Benchmark binary not found, building it first ..."
    "${SCRIPT_DIR}/build.sh"
fi

# ---- build the mpirun / benchmark command --------------------------------
mkdir -p "${OUTDIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NAME_TAG="ccl_bench_${BACKEND}_${DEVICES}dev${TAG:+_${TAG}}_${TIMESTAMP}"
CSV_FILE="${OUTDIR}/${NAME_TAG}.csv"
LOG_FILE="${OUTDIR}/${NAME_TAG}.log"

BENCH_ARGS=(-l "${OPS}" -y "${SIZES}" -d "${DTYPE}" -i "${ITERS}" -w "${WARMUP_ITERS}" -o "${CSV_FILE}")

if [[ "${BACKEND}" == "xpu" ]]; then
    BENCH_ARGS=(-b sycl -a gpu "${BENCH_ARGS[@]}")
    RUN_CMD=("${SCRIPT_DIR}/bind_gpu.sh" "${BIN}" "${BENCH_ARGS[@]}")
else
    BENCH_ARGS=(-b host "${BENCH_ARGS[@]}")
    RUN_CMD=("${BIN}" "${BENCH_ARGS[@]}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${EXTRA_ARGS})
    RUN_CMD+=("${EXTRA_ARR[@]}")
fi

MPI_ARGS=(-n "${DEVICES}" -ppn "${PPN}")
[[ -n "${HOSTFILE}" ]] && MPI_ARGS+=(-f "${HOSTFILE}")

# The benchmark tool truncates csv_filepath and writes its own header row.
echo "=== Running: mpirun ${MPI_ARGS[*]} ${RUN_CMD[*]} ==="
mpirun "${MPI_ARGS[@]}" "${RUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo
echo "=== Log saved to: ${LOG_FILE} ==="
echo "=== CSV saved to: ${CSV_FILE} ==="
echo
echo "=== CSV summary ==="
if command -v column >/dev/null 2>&1; then
    column -t -s, "${CSV_FILE}"
else
    cat "${CSV_FILE}"
fi
