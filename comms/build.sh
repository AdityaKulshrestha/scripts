#!/usr/bin/env bash
#
# Builds the "benchmark" tool that ships with the installed oneCCL/oneAPI
# distribution ($CCL_ROOT/share/doc/ccl/examples/benchmark). It is built once
# with SYCL enabled so the same binary can benchmark both the "host" (CPU)
# and "sycl" (XPU/GPU) backends.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if [[ -z "${CCL_ROOT:-}" ]]; then
    echo "ERROR: CCL_ROOT is not set. Source the oneAPI environment first, e.g.:" >&2
    echo "  source /swtools/intel/oneapi/setvars.sh" >&2
    exit 1
fi

if [[ -z "${I_MPI_ROOT:-}" ]]; then
    echo "ERROR: I_MPI_ROOT is not set (Intel MPI not found in the environment)." >&2
    echo "  Source the oneAPI environment first, e.g.: source /swtools/intel/oneapi/setvars.sh" >&2
    exit 1
fi

EXAMPLES_SRC="${CCL_ROOT}/share/doc/ccl/examples"
if [[ ! -d "${EXAMPLES_SRC}/benchmark" ]]; then
    echo "ERROR: could not find oneCCL benchmark sources under: ${EXAMPLES_SRC}" >&2
    exit 1
fi

# Route mpicc/mpicxx through the SYCL-capable Intel compiler so the sycl
# (GPU) backend of the benchmark tool gets compiled in.
export I_MPI_CC="${I_MPI_CC:-icx}"
export I_MPI_CXX="${I_MPI_CXX:-icpx}"

echo "Configuring build in ${BUILD_DIR} (source: ${EXAMPLES_SRC}) ..."
cmake -S "${EXAMPLES_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCOMPUTE_BACKEND=dpcpp \
    -DCMAKE_BUILD_TYPE=Release

echo "Building 'benchmark' target ..."
cmake --build "${BUILD_DIR}" --target benchmark -j"$(nproc)"

BIN="${BUILD_DIR}/benchmark/benchmark"
if [[ -x "${BIN}" ]]; then
    echo "Build OK: ${BIN}"
else
    echo "ERROR: build finished but binary not found at ${BIN}" >&2
    exit 1
fi
