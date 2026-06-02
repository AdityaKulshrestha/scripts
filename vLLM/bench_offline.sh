#!/bin/bash

# =============================================================================
# vLLM Offline Latency Benchmark Script
# Runs vllm bench latency with sweep over input/output tokens and batch sizes
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
MODEL=""
INPUT_TOKENS="128,1024"
OUTPUT_TOKENS="1,32"
BATCH_SIZES="1,4,8,16"
TP=1
NUM_ITERS=10
NUM_WARMUP=5
MAX_MODEL_LEN=8192
RESULTS_DIR="./results"
RESULT_FILENAME=""
INTERACTIVE=false
PROFILE=false
PROFILE_DIR="./vllm_profile"

# =============================================================================
# Logging
# =============================================================================
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }

# =============================================================================
# Help
# =============================================================================
show_help() {
    cat << 'EOF'
================================================================================
                  vLLM Offline Latency Benchmark Script
================================================================================

USAGE:
    ./bench_offline.sh --model <model_name> [OPTIONS]
    ./bench_offline.sh -i                    # Interactive mode

DESCRIPTION:
    Runs vLLM latency benchmarks with a sweep over input/output tokens and
    batch sizes. Calculates Inter-Token Latency (ITL) from paired runs.

DEFAULT SWEEP CONFIGURATION:
    Input Tokens:   128, 1024
    Output Tokens:  1, 32
    Batch Sizes:    1, 4, 8, 16

    ITL Calculation:
    - For 128 input:  ITL = (latency_128_32 - latency_128_1) / (32 - 1)
    - For 1024 input: ITL = (latency_1024_32 - latency_1024_1) / (32 - 1)

REQUIRED OPTIONS:
    --model <name>              Model name/path (e.g., google/gemma-3-4b-it)

OPTIONAL OPTIONS:
    --input-tokens <list>       Comma-separated input token lengths (default: 128,1024)
    --output-tokens <list>      Comma-separated output token lengths (default: 1,32)
    --batch-sizes <list>        Comma-separated batch sizes (default: 1,4,8,16)
    --tp <num>                  Tensor parallelism (default: 1)
    --num-iters <num>           Number of benchmark iterations (default: 10)
    --num-warmup <num>          Number of warmup iterations (default: 5)
    --max-model-len <num>       Max model context length (default: 8192)
    --results-dir <path>        Results directory (default: ./results)
    --result-filename <name>    Custom result filename (without extension)
    -i, --interactive           Interactive mode
    -p, --profile               Enable PyTorch profiling
    --profile-dir <path>        Profile output directory (default: ./vllm_profile)
    -h, --help                  Show this help

EXAMPLES:
    # Quick run with defaults
    ./bench_offline.sh --model google/gemma-3-4b-it

    # Interactive mode
    ./bench_offline.sh -i

    # With profiling
    ./bench_offline.sh --model meta-llama/Llama-2-7b -p

    # Custom sweep
    ./bench_offline.sh --model google/gemma-3-4b-it \
        --input-tokens 256,512,1024 --output-tokens 1,16,32 \
        --batch-sizes 1,8

================================================================================
EOF
    exit 0
}

# =============================================================================
# Environment Check
# =============================================================================
check_vllm_env() {
    log_info "Checking for vLLM environment..."

    # First, check if vLLM is already available as a Python module (env already activated)
    if python -c "import vllm; print(vllm.__version__)" &>/dev/null; then
        VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        log_success "vLLM Python module found in current environment: $VLLM_VER"
        return 0
    fi

    # Check if vllm CLI command is available
    if command -v vllm &>/dev/null; then
        VLLM_VER=$(vllm --version 2>/dev/null || echo "unknown")
        log_success "vLLM CLI found in current environment: $VLLM_VER"
        return 0
    fi

    # Check .venv if it exists
    if [[ -d ".venv" ]]; then
        log_info "Found .venv, activating..."
        source .venv/bin/activate
        if python -c "import vllm; print(vllm.__version__)" &>/dev/null; then
            VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
            log_success "vLLM found in .venv: $VLLM_VER"
            return 0
        fi
    fi

    # No vLLM found
    log_error "vLLM not found in current environment or .venv"
    echo ""
    echo "Please create a vLLM environment first:"
    echo "  ./setup_vllm.sh -d cpu -m precompiled"
    echo ""
    echo "Or activate your existing vLLM environment before running this script."
    exit 1
}

# =============================================================================
# Interactive Mode
# =============================================================================
run_interactive() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  vLLM Offline Benchmark - Interactive${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Model
    read -p "Model name (required): " MODEL
    [[ -z "$MODEL" ]] && { log_error "Model is required"; exit 1; }

    # Input tokens
    read -p "Input tokens [128,1024]: " input
    INPUT_TOKENS="${input:-128,1024}"

    # Output tokens
    read -p "Output tokens [1,32]: " input
    OUTPUT_TOKENS="${input:-1,32}"

    # Batch sizes
    read -p "Batch sizes [1,4,8,16]: " input
    BATCH_SIZES="${input:-1,4,8,16}"

    # Tensor parallelism
    read -p "Tensor parallelism [1]: " input
    TP="${input:-1}"

    # Iterations
    read -p "Number of iterations [10]: " input
    NUM_ITERS="${input:-10}"

    # Warmup
    read -p "Warmup iterations [5]: " input
    NUM_WARMUP="${input:-5}"

    # Max model length
    read -p "Max model length [8192]: " input
    MAX_MODEL_LEN="${input:-8192}"

    # Results directory
    read -p "Results directory [./results]: " input
    RESULTS_DIR="${input:-./results}"

    # Profiling
    read -p "Enable profiling? (y/n) [n]: " input
    if [[ "${input,,}" == "y" ]]; then
        PROFILE=true
        read -p "Profile directory [./vllm_profile]: " input
        PROFILE_DIR="${input:-./vllm_profile}"
    fi

    echo ""
}

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) show_help ;;
            -i|--interactive) INTERACTIVE=true; shift ;;
            -p|--profile) PROFILE=true; shift ;;
            --model) MODEL="$2"; shift 2 ;;
            --input-tokens) INPUT_TOKENS="$2"; shift 2 ;;
            --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
            --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
            --tp) TP="$2"; shift 2 ;;
            --num-iters) NUM_ITERS="$2"; shift 2 ;;
            --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
            --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
            --results-dir) RESULTS_DIR="$2"; shift 2 ;;
            --result-filename) RESULT_FILENAME="$2"; shift 2 ;;
            --profile-dir) PROFILE_DIR="$2"; shift 2 ;;
            *) log_error "Unknown option: $1"; show_help ;;
        esac
    done
}

# =============================================================================
# Validate Arguments
# =============================================================================
validate_args() {
    if [[ -z "$MODEL" ]]; then
        log_error "Missing required argument: --model"
        echo ""
        echo "Use --help for usage or -i for interactive mode"
        exit 1
    fi
}

# =============================================================================
# Run Single Benchmark
# =============================================================================
run_single_benchmark() {
    local input_len="$1"
    local output_len="$2"
    local batch_size="$3"

    local cmd="vllm bench latency"
    cmd+=" --model $MODEL"
    cmd+=" --input-len $input_len"
    cmd+=" --output-len $output_len"
    cmd+=" --batch-size $batch_size"
    cmd+=" --num-iters $NUM_ITERS"
    cmd+=" --num-iters-warmup $NUM_WARMUP"
    cmd+=" --max-model-len $MAX_MODEL_LEN"
    cmd+=" -tp $TP"
    cmd+=" --no-enable-prefix-caching"
    cmd+=" --trust-remote-code"

    # Add profiling if enabled
    if $PROFILE; then
        local profile_subdir="${PROFILE_DIR}/in${input_len}_out${output_len}_bs${batch_size}"
        mkdir -p "$profile_subdir"
        cmd+=" --profile --profiler-config '{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$profile_subdir\"}'"
    fi

    echo "$cmd"
}

# =============================================================================
# Extract Latency from Output
# =============================================================================
extract_latency() {
    local output="$1"
    local metric="$2"  # avg, p10, p25, p50, p75, p90, p99

    case "$metric" in
        avg)    echo "$output" | grep -E "^Avg latency:" | awk '{print $3}' ;;
        p10)    echo "$output" | grep -E "^10% percentile" | awk '{print $4}' ;;
        p25)    echo "$output" | grep -E "^25% percentile" | awk '{print $4}' ;;
        p50)    echo "$output" | grep -E "^50% percentile" | awk '{print $4}' ;;
        p75)    echo "$output" | grep -E "^75% percentile" | awk '{print $4}' ;;
        p90)    echo "$output" | grep -E "^90% percentile" | awk '{print $4}' ;;
        p99)    echo "$output" | grep -E "^99% percentile" | awk '{print $4}' ;;
    esac
}

# =============================================================================
# Calculate ITL
# =============================================================================
calculate_itl() {
    local latency_high="$1"  # e.g., latency for 32 output tokens
    local latency_low="$2"   # e.g., latency for 1 output token
    local tokens_high="$3"   # e.g., 32
    local tokens_low="$4"    # e.g., 1

    if [[ -n "$latency_high" && -n "$latency_low" ]]; then
        local token_diff=$((tokens_high - tokens_low))
        if [[ $token_diff -gt 0 ]]; then
            # ITL in milliseconds
            echo "scale=6; ($latency_high - $latency_low) / $token_diff * 1000" | bc 2>/dev/null || echo "N/A"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# =============================================================================
# Run Benchmark Sweep
# =============================================================================
run_benchmark() {
    mkdir -p "$RESULTS_DIR"
    $PROFILE && mkdir -p "$PROFILE_DIR"

    # Generate result filename
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local model_short
    model_short=$(basename "$MODEL" | tr '/' '_')

    if [[ -n "$RESULT_FILENAME" ]]; then
        CSV_FILE="${RESULTS_DIR}/${RESULT_FILENAME}.csv"
    else
        CSV_FILE="${RESULTS_DIR}/bench_offline_${model_short}_${timestamp}.csv"
    fi

    LOG_FILE="${RESULTS_DIR}/bench_offline_${model_short}_${timestamp}.log"

    # Convert to arrays
    IFS=',' read -ra INPUT_ARR <<< "$INPUT_TOKENS"
    IFS=',' read -ra OUTPUT_ARR <<< "$OUTPUT_TOKENS"
    IFS=',' read -ra BATCH_ARR <<< "$BATCH_SIZES"

    # Print configuration
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}      Offline Benchmark Configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "  Model:          $MODEL"
    echo "  Input Tokens:   ${INPUT_ARR[*]}"
    echo "  Output Tokens:  ${OUTPUT_ARR[*]}"
    echo "  Batch Sizes:    ${BATCH_ARR[*]}"
    echo "  TP:             $TP"
    echo "  Iterations:     $NUM_ITERS (warmup: $NUM_WARMUP)"
    echo "  Max Model Len:  $MAX_MODEL_LEN"
    echo "  Results:        $CSV_FILE"
    $PROFILE && echo "  Profile Dir:    $PROFILE_DIR"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Calculate total runs
    local total_runs=$((${#INPUT_ARR[@]} * ${#OUTPUT_ARR[@]} * ${#BATCH_ARR[@]}))
    local current_run=0

    # Store all metrics for summary output
    declare -A LATENCY_MAP
    declare -A P10_MAP
    declare -A P25_MAP
    declare -A P50_MAP
    declare -A P75_MAP
    declare -A P90_MAP
    declare -A P99_MAP

    log_info "Starting benchmark sweep ($total_runs configurations)..."
    echo ""

    # Run sweep
    for input_len in "${INPUT_ARR[@]}"; do
        for output_len in "${OUTPUT_ARR[@]}"; do
            for batch_size in "${BATCH_ARR[@]}"; do
                current_run=$((current_run + 1))

                echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo -e "Run $current_run/$total_runs: input=$input_len, output=$output_len, batch=$batch_size"
                echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

                # Build and execute command
                local cmd
                cmd=$(run_single_benchmark "$input_len" "$output_len" "$batch_size")

                echo "Command: $cmd" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"

                local output
                local exit_code=0
                output=$(eval "$cmd" 2>&1) || exit_code=$?

                echo "$output" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"

                local key="${input_len}_${output_len}_${batch_size}"

                # Check success
                if ! echo "$output" | grep -q "Avg latency:"; then
                    log_error "Benchmark failed for input=$input_len, output=$output_len, batch=$batch_size"
                    echo "$output" | tail -10
                    LATENCY_MAP["$key"]="ERROR"
                    P10_MAP["$key"]="ERROR"
                    P25_MAP["$key"]="ERROR"
                    P50_MAP["$key"]="ERROR"
                    P75_MAP["$key"]="ERROR"
                    P90_MAP["$key"]="ERROR"
                    P99_MAP["$key"]="ERROR"
                    continue
                fi

                # Extract metrics
                local avg_lat p10 p25 p50 p75 p90 p99
                avg_lat=$(extract_latency "$output" "avg")
                p10=$(extract_latency "$output" "p10")
                p25=$(extract_latency "$output" "p25")
                p50=$(extract_latency "$output" "p50")
                p75=$(extract_latency "$output" "p75")
                p90=$(extract_latency "$output" "p90")
                p99=$(extract_latency "$output" "p99")

                # Store all metrics
                LATENCY_MAP["$key"]="$avg_lat"
                P10_MAP["$key"]="$p10"
                P25_MAP["$key"]="$p25"
                P50_MAP["$key"]="$p50"
                P75_MAP["$key"]="$p75"
                P90_MAP["$key"]="$p90"
                P99_MAP["$key"]="$p99"

                log_success "Avg latency: ${avg_lat}s | P50: ${p50}s | P90: ${p90}s"
                echo ""
            done
        done
    done

    # Find output token pairs for ITL calculation
    local out_low out_high
    out_low=$(echo "${OUTPUT_ARR[*]}" | tr ' ' '\n' | sort -n | head -1)
    out_high=$(echo "${OUTPUT_ARR[*]}" | tr ' ' '\n' | sort -n | tail -1)

    # Write summary CSV grouped by input tokens
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}         Generating Summary CSV${NC}"
    echo -e "${CYAN}========================================${NC}"

    # Main CSV header (raw data)
    echo "timestamp,model,input_len,output_len,batch_size,tp,avg_latency_s,p10_s,p25_s,p50_s,p75_s,p90_s,p99_s" > "$CSV_FILE"

    # ITL CSV with cross-reference columns
    ITL_FILE="${CSV_FILE%.csv}_itl.csv"
    echo "timestamp,model,input_len,batch_size,tp,output_low,output_high,avg_latency_out_${out_low}_s,avg_latency_out_${out_high}_s,itl_ms" > "$ITL_FILE"

    local run_timestamp
    run_timestamp=$(date -Iseconds)

    for input_len in "${INPUT_ARR[@]}"; do
        echo ""
        echo -e "${YELLOW}Input Tokens: $input_len${NC}"
        echo "─────────────────────────────────────────"

        for output_len in "${OUTPUT_ARR[@]}"; do
            echo -e "  ${BLUE}Output Tokens: $output_len${NC}"

            for batch_size in "${BATCH_ARR[@]}"; do
                local key="${input_len}_${output_len}_${batch_size}"
                local avg_lat="${LATENCY_MAP[$key]:-N/A}"
                local p10="${P10_MAP[$key]:-N/A}"
                local p25="${P25_MAP[$key]:-N/A}"
                local p50="${P50_MAP[$key]:-N/A}"
                local p75="${P75_MAP[$key]:-N/A}"
                local p90="${P90_MAP[$key]:-N/A}"
                local p99="${P99_MAP[$key]:-N/A}"

                # Write to main CSV
                echo "$run_timestamp,$MODEL,$input_len,$output_len,$batch_size,$TP,$avg_lat,$p10,$p25,$p50,$p75,$p90,$p99" >> "$CSV_FILE"

                # Display
                if [[ "$avg_lat" != "ERROR" && "$avg_lat" != "N/A" ]]; then
                    printf "    Batch %2d: Avg=%.4fs P50=%.4fs P90=%.4fs\n" "$batch_size" "$avg_lat" "$p50" "$p90"
                else
                    printf "    Batch %2d: ERROR\n" "$batch_size"
                fi
            done
        done
    done

    # ITL Summary Section
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}     Inter-Token Latency (ITL) Summary${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "Formula: ITL = (latency_out${out_high} - latency_out${out_low}) / ($out_high - $out_low)"
    echo ""

    if [[ "$out_low" != "$out_high" ]]; then
        printf "%-12s %-12s %-15s %-15s %-12s\n" "Input Len" "Batch Size" "Lat(out=$out_low)" "Lat(out=$out_high)" "ITL (ms)"
        printf "%-12s %-12s %-15s %-15s %-12s\n" "─────────" "──────────" "──────────────" "──────────────" "─────────"

        for input_len in "${INPUT_ARR[@]}"; do
            for batch_size in "${BATCH_ARR[@]}"; do
                local lat_low="${LATENCY_MAP[${input_len}_${out_low}_${batch_size}]:-N/A}"
                local lat_high="${LATENCY_MAP[${input_len}_${out_high}_${batch_size}]:-N/A}"

                local itl="N/A"
                if [[ "$lat_low" != "N/A" && "$lat_high" != "N/A" && "$lat_low" != "ERROR" && "$lat_high" != "ERROR" ]]; then
                    itl=$(calculate_itl "$lat_high" "$lat_low" "$out_high" "$out_low")
                fi

                # Write to ITL CSV with all cross-reference columns
                echo "$run_timestamp,$MODEL,$input_len,$batch_size,$TP,$out_low,$out_high,$lat_low,$lat_high,$itl" >> "$ITL_FILE"

                # Display
                if [[ "$lat_low" != "ERROR" && "$lat_high" != "ERROR" ]]; then
                    printf "%-12s %-12s %-15.4f %-15.4f %-12s\n" "$input_len" "$batch_size" "$lat_low" "$lat_high" "$itl"
                else
                    printf "%-12s %-12s %-15s %-15s %-12s\n" "$input_len" "$batch_size" "$lat_low" "$lat_high" "N/A"
                fi
            done
        done
    else
        log_warn "Cannot calculate ITL: need at least 2 different output token lengths"
    fi

    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Summary
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}            Benchmark Complete${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "  Results CSV:    $CSV_FILE"
    echo "  ITL CSV:        $ITL_FILE"
    echo "  Log file:       $LOG_FILE"
    $PROFILE && echo "  Profile data:   $PROFILE_DIR"
    echo "  Total runs:     $current_run"
    echo -e "${CYAN}========================================${NC}"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   vLLM Offline Latency Benchmark Tool  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"

    parse_args "$@"

    # Interactive mode
    if $INTERACTIVE; then
        run_interactive
    fi

    # Validate
    validate_args

    # Check environment
    check_vllm_env

    # Run benchmark
    run_benchmark

    echo ""
    log_success "Benchmark complete!"
}

main "$@"
