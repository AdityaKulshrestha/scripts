#!/bin/bash

# =============================================================================
# vLLM Online Benchmark Script
# Supports: Interactive & Command-line modes
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
BASE_URL="http://localhost:8000"
MODEL=""
INPUT_TOKENS=""
OUTPUT_TOKENS=""
CONCURRENCY=""
NUM_PROMPTS=""
REQUEST_RATE="inf"
DATASET="random"
RESULTS_DIR="./results"
RESULT_FILENAME=""
APPEND_RESULT=false
INTERACTIVE=false

# Profiling defaults
PROFILE=false
PROFILE_DIR="./vllm_profile"
PROFILE_SHAPES=false
PROFILE_MEMORY=false
PROFILE_STACK=true
PROFILE_FLOPS=false

# Visualization
PLOT_TIMELINE=false

# Extra args for vllm bench serve
BENCH_EXTRA_ARGS=()

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
                    vLLM Online Benchmark Script
================================================================================

USAGE:
    ./run_benchmark.sh [OPTIONS]
    ./run_benchmark.sh --interactive

MODES:
    Command-line    Provide all options via arguments
    Interactive     Prompt for options (use --interactive or -i)

REQUIRED OPTIONS (command-line mode):
    --model <name>              Model name/path
    --input-tokens <num>        Input token length
    --output-tokens <num>       Output token length
    --concurrency <num>         Max concurrent requests
    --num-prompts <num>         Total number of prompts

OPTIONAL OPTIONS:
    --base-url <url>            Server URL (default: http://localhost:8000)
    --request-rate <rate>       Request rate (default: inf)
    --dataset <name>            Dataset: random, sharegpt, sonnet (default: random)
    --results-dir <path>        Results directory (default: ./results)
    --result-filename <name>    Custom result filename (without extension)
    --append-result             Append to existing result file
    -i, --interactive           Interactive mode
    -h, --help                  Show this help

PROFILING OPTIONS:
    --profile                   Enable PyTorch profiling
    --profile-dir <path>        Profile output directory (default: ./vllm_profile)
    --profile-shapes            Record tensor shapes
    --profile-memory            Record memory usage
    --profile-stack             Record stack info (default: on)
    --profile-flops             Record FLOPs

VISUALIZATION:
    --plot-timeline             Generate timeline plot after benchmark

PASS-THROUGH:
    --bench-args <args...>      Additional args for vllm bench serve (must be last)

EXAMPLES:
    # Command-line mode
    ./run_benchmark.sh --model meta-llama/Llama-2-7b \
        --input-tokens 128 --output-tokens 128 \
        --concurrency 4 --num-prompts 16

    # Interactive mode
    ./run_benchmark.sh -i

    # With profiling
    ./run_benchmark.sh --model google/gemma-2b \
        --input-tokens 256 --output-tokens 256 \
        --concurrency 8 --num-prompts 32 \
        --profile --profile-flops

    # With visualization
    ./run_benchmark.sh --model meta-llama/Llama-2-7b \
        --input-tokens 128 --output-tokens 128 \
        --concurrency 4 --num-prompts 16 \
        --plot-timeline

================================================================================
EOF
    exit 0
}

# =============================================================================
# Environment Check
# =============================================================================
check_vllm_env() {
    log_info "Checking for vLLM environment..."

    # Check current environment
    if command -v vllm &>/dev/null; then
        VLLM_VER=$(vllm --version 2>/dev/null || echo "unknown")
        log_success "vLLM found in current environment: $VLLM_VER"
        return 0
    fi

    # Check .venv
    if [[ -d ".venv" ]]; then
        log_info "Found .venv, activating..."
        source .venv/bin/activate
        if command -v vllm &>/dev/null; then
            VLLM_VER=$(vllm --version 2>/dev/null || echo "unknown")
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
# Server Health Check
# =============================================================================
check_server() {
    log_info "Checking server at $BASE_URL..."
    
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")

    if [[ "$status" == "200" ]]; then
        log_success "Server is ready"
        return 0
    else
        log_error "Server not ready at $BASE_URL (HTTP: $status)"
        echo ""
        echo "Please ensure the vLLM server is running:"
        echo "  vllm serve <model> --host 0.0.0.0 --port 8000"
        exit 1
    fi
}

# =============================================================================
# Interactive Mode
# =============================================================================
run_interactive() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}    vLLM Benchmark - Interactive Mode${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Server URL
    read -p "Server URL [http://localhost:8000]: " input
    BASE_URL="${input:-http://localhost:8000}"

    # Model
    read -p "Model name (required): " MODEL
    [[ -z "$MODEL" ]] && { log_error "Model is required"; exit 1; }

    # Input tokens
    read -p "Input tokens (required): " INPUT_TOKENS
    [[ -z "$INPUT_TOKENS" ]] && { log_error "Input tokens required"; exit 1; }

    # Output tokens
    read -p "Output tokens (required): " OUTPUT_TOKENS
    [[ -z "$OUTPUT_TOKENS" ]] && { log_error "Output tokens required"; exit 1; }

    # Concurrency
    read -p "Concurrency (required): " CONCURRENCY
    [[ -z "$CONCURRENCY" ]] && { log_error "Concurrency required"; exit 1; }

    # Num prompts
    read -p "Number of prompts (required): " NUM_PROMPTS
    [[ -z "$NUM_PROMPTS" ]] && { log_error "Num prompts required"; exit 1; }

    # Request rate
    read -p "Request rate [inf]: " input
    REQUEST_RATE="${input:-inf}"

    # Dataset
    echo "Dataset options: random, sharegpt, sonnet"
    read -p "Dataset [random]: " input
    DATASET="${input:-random}"

    # Results directory
    read -p "Results directory [./results]: " input
    RESULTS_DIR="${input:-./results}"

    # Profiling
    read -p "Enable profiling? (y/n) [n]: " input
    if [[ "${input,,}" == "y" ]]; then
        PROFILE=true
        read -p "Profile directory [./vllm_profile]: " input
        PROFILE_DIR="${input:-./vllm_profile}"
        
        read -p "Record tensor shapes? (y/n) [n]: " input
        [[ "${input,,}" == "y" ]] && PROFILE_SHAPES=true
        
        read -p "Record memory? (y/n) [n]: " input
        [[ "${input,,}" == "y" ]] && PROFILE_MEMORY=true
        
        read -p "Record FLOPs? (y/n) [n]: " input
        [[ "${input,,}" == "y" ]] && PROFILE_FLOPS=true
    fi

    # Visualization
    read -p "Generate timeline plot? (y/n) [n]: " input
    [[ "${input,,}" == "y" ]] && PLOT_TIMELINE=true

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
            --base-url) BASE_URL="$2"; shift 2 ;;
            --model) MODEL="$2"; shift 2 ;;
            --input-tokens) INPUT_TOKENS="$2"; shift 2 ;;
            --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
            --concurrency) CONCURRENCY="$2"; shift 2 ;;
            --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
            --request-rate) REQUEST_RATE="$2"; shift 2 ;;
            --dataset) DATASET="$2"; shift 2 ;;
            --results-dir) RESULTS_DIR="$2"; shift 2 ;;
            --result-filename) RESULT_FILENAME="$2"; shift 2 ;;
            --append-result) APPEND_RESULT=true; shift ;;
            --profile) PROFILE=true; shift ;;
            --profile-dir) PROFILE_DIR="$2"; shift 2 ;;
            --profile-shapes) PROFILE_SHAPES=true; shift ;;
            --profile-memory) PROFILE_MEMORY=true; shift ;;
            --profile-stack) PROFILE_STACK=true; shift ;;
            --profile-flops) PROFILE_FLOPS=true; shift ;;
            --plot-timeline) PLOT_TIMELINE=true; shift ;;
            --bench-args)
                shift
                BENCH_EXTRA_ARGS=("$@")
                break
                ;;
            *) log_error "Unknown option: $1"; show_help ;;
        esac
    done
}

# =============================================================================
# Validate Arguments
# =============================================================================
validate_args() {
    local missing=()
    
    [[ -z "$MODEL" ]] && missing+=("--model")
    [[ -z "$INPUT_TOKENS" ]] && missing+=("--input-tokens")
    [[ -z "$OUTPUT_TOKENS" ]] && missing+=("--output-tokens")
    [[ -z "$CONCURRENCY" ]] && missing+=("--concurrency")
    [[ -z "$NUM_PROMPTS" ]] && missing+=("--num-prompts")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required arguments:"
        printf '  %s\n' "${missing[@]}"
        echo ""
        echo "Use --help for usage or -i for interactive mode"
        exit 1
    fi
}

# =============================================================================
# Build Benchmark Command
# =============================================================================
build_bench_cmd() {
    local cmd="vllm bench serve"
    cmd+=" --backend openai-chat"
    cmd+=" --base-url $BASE_URL"
    cmd+=" --model $MODEL"
    cmd+=" --endpoint /v1/chat/completions"
    cmd+=" --ignore-eos"
    cmd+=" --metric-percentiles 90"

    # Dataset
    case "$DATASET" in
        random)
            cmd+=" --dataset-name random"
            cmd+=" --random-input-len $INPUT_TOKENS"
            cmd+=" --random-output-len $OUTPUT_TOKENS"
            ;;
        sharegpt)
            cmd+=" --dataset-name sharegpt"
            ;;
        sonnet)
            cmd+=" --dataset-name sonnet"
            cmd+=" --sonnet-input-len $INPUT_TOKENS"
            cmd+=" --sonnet-output-len $OUTPUT_TOKENS"
            cmd+=" --sonnet-prefix-len 100"
            ;;
    esac

    cmd+=" --request-rate $REQUEST_RATE"
    cmd+=" --num-prompts $NUM_PROMPTS"
    cmd+=" --max-concurrency $CONCURRENCY"

    # Profiling
    if $PROFILE; then
        cmd+=" --profiler torch"
        cmd+=" --torch-profiler-dir $PROFILE_DIR"
        $PROFILE_SHAPES && cmd+=" --torch-profiler-record-shapes"
        $PROFILE_MEMORY && cmd+=" --torch-profiler-with-memory"
        $PROFILE_STACK && cmd+=" --torch-profiler-with-stack"
        $PROFILE_FLOPS && cmd+=" --torch-profiler-with-flops"
    fi

    # Extra args
    if [[ ${#BENCH_EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=" ${BENCH_EXTRA_ARGS[*]}"
    fi

    echo "$cmd"
}

# =============================================================================
# Run Benchmark
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
        RESULT_FILE="${RESULTS_DIR}/${RESULT_FILENAME}.json"
    else
        RESULT_FILE="${RESULTS_DIR}/benchmark_${model_short}_${timestamp}.json"
    fi

    local LOG_FILE="${RESULTS_DIR}/benchmark_${model_short}_${timestamp}.log"

    # Print configuration
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}        Benchmark Configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "  Server:       $BASE_URL"
    echo "  Model:        $MODEL"
    echo "  Input:        $INPUT_TOKENS tokens"
    echo "  Output:       $OUTPUT_TOKENS tokens"
    echo "  Concurrency:  $CONCURRENCY"
    echo "  Prompts:      $NUM_PROMPTS"
    echo "  Request Rate: $REQUEST_RATE"
    echo "  Dataset:      $DATASET"
    echo "  Results:      $RESULT_FILE"
    $PROFILE && echo "  Profile Dir:  $PROFILE_DIR"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Build command
    local cmd
    cmd=$(build_bench_cmd)

    log_info "Running benchmark..."
    echo "Command: $cmd"
    echo ""

    # Execute benchmark
    local output
    local exit_code=0
    output=$(eval "$cmd" 2>&1) || exit_code=$?

    # Log output
    echo "$output" > "$LOG_FILE"

    # Check success
    if ! echo "$output" | grep -q "Serving Benchmark Result"; then
        log_error "Benchmark failed"
        echo "$output" | tail -20
        exit 1
    fi

    log_success "Benchmark completed"
    echo ""

    # Extract and save results as JSON
    save_results_json "$output"

    # Display key metrics
    echo ""
    echo -e "${CYAN}Key Metrics:${NC}"
    echo "$output" | grep -E "(Mean TTFT|Mean TPOT|Mean ITL|Request throughput|Output token throughput)" | head -10
    echo ""

    # Generate timeline plot if requested
    if $PLOT_TIMELINE; then
        generate_timeline_plot
    fi

    echo ""
    log_success "Results saved to: $RESULT_FILE"
    echo "Log saved to: $LOG_FILE"
}

# =============================================================================
# Save Results as JSON
# =============================================================================
save_results_json() {
    local output="$1"
    
    # Extract metrics
    local mean_ttft median_ttft p90_ttft
    local mean_tpot median_tpot p90_tpot
    local mean_itl median_itl p90_itl
    local req_throughput output_throughput

    mean_ttft=$(echo "$output" | grep -E "^Mean TTFT \(ms\):" | awk '{print $4}' || echo "null")
    median_ttft=$(echo "$output" | grep -E "^Median TTFT \(ms\):" | awk '{print $4}' || echo "null")
    p90_ttft=$(echo "$output" | grep -E "^P90 TTFT \(ms\):" | awk '{print $4}' || echo "null")

    mean_tpot=$(echo "$output" | grep -E "^Mean TPOT \(ms\):" | awk '{print $4}' || echo "null")
    median_tpot=$(echo "$output" | grep -E "^Median TPOT \(ms\):" | awk '{print $4}' || echo "null")
    p90_tpot=$(echo "$output" | grep -E "^P90 TPOT \(ms\):" | awk '{print $4}' || echo "null")

    mean_itl=$(echo "$output" | grep -E "^Mean ITL \(ms\):" | awk '{print $4}' || echo "null")
    median_itl=$(echo "$output" | grep -E "^Median ITL \(ms\):" | awk '{print $4}' || echo "null")
    p90_itl=$(echo "$output" | grep -E "^P90 ITL \(ms\):" | awk '{print $4}' || echo "null")

    req_throughput=$(echo "$output" | grep -E "^Request throughput \(req/s\):" | awk '{print $4}' || echo "null")
    output_throughput=$(echo "$output" | grep -E "^Output token throughput \(tok/s\):" | awk '{print $5}' || echo "null")

    # Calculate interactivity
    local interactivity="null"
    if [[ "$output_throughput" != "null" && "$CONCURRENCY" -gt 0 ]]; then
        interactivity=$(echo "scale=2; $output_throughput / $CONCURRENCY" | bc 2>/dev/null || echo "null")
    fi

    # Build JSON
    local json_result
    json_result=$(cat <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "config": {
    "model": "$MODEL",
    "base_url": "$BASE_URL",
    "input_tokens": $INPUT_TOKENS,
    "output_tokens": $OUTPUT_TOKENS,
    "concurrency": $CONCURRENCY,
    "num_prompts": $NUM_PROMPTS,
    "request_rate": "$REQUEST_RATE",
    "dataset": "$DATASET"
  },
  "metrics": {
    "ttft_ms": {
      "mean": ${mean_ttft:-null},
      "median": ${median_ttft:-null},
      "p90": ${p90_ttft:-null}
    },
    "tpot_ms": {
      "mean": ${mean_tpot:-null},
      "median": ${median_tpot:-null},
      "p90": ${p90_tpot:-null}
    },
    "itl_ms": {
      "mean": ${mean_itl:-null},
      "median": ${median_itl:-null},
      "p90": ${p90_itl:-null}
    },
    "throughput": {
      "requests_per_sec": ${req_throughput:-null},
      "output_tokens_per_sec": ${output_throughput:-null},
      "tokens_per_sec_per_user": ${interactivity:-null}
    }
  },
  "profiling": {
    "enabled": $PROFILE,
    "directory": $(if $PROFILE; then echo "\"$PROFILE_DIR\""; else echo "null"; fi)
  }
}
EOF
)

    # Handle append mode
    if $APPEND_RESULT && [[ -f "$RESULT_FILE" ]]; then
        # Read existing JSON array and append
        local existing
        existing=$(cat "$RESULT_FILE")
        if echo "$existing" | grep -q '^\['; then
            # Remove trailing ] and append new result
            existing="${existing%]}"
            echo "${existing},${json_result}]" > "$RESULT_FILE"
        else
            # Convert single object to array
            echo "[${existing},${json_result}]" > "$RESULT_FILE"
        fi
    else
        echo "$json_result" > "$RESULT_FILE"
    fi
}

# =============================================================================
# Generate Timeline Plot
# =============================================================================
generate_timeline_plot() {
    log_info "Generating timeline plot..."

    # Check if matplotlib is available
    if ! python3 -c "import matplotlib" 2>/dev/null; then
        log_warn "matplotlib not installed. Skipping plot generation."
        echo "  Install with: pip install matplotlib"
        return
    fi

    local plot_file="${RESULTS_DIR}/timeline_$(date +%Y%m%d_%H%M%S).png"

    python3 << EOF
import json
import matplotlib.pyplot as plt

try:
    with open("$RESULT_FILE", 'r') as f:
        data = json.load(f)
    
    # Handle both single result and array
    if isinstance(data, list):
        data = data[-1]  # Use latest result
    
    metrics = data['metrics']
    config = data['config']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Benchmark Results: {config['model']}", fontsize=14)
    
    # TTFT
    ttft = metrics['ttft_ms']
    axes[0].bar(['Mean', 'Median', 'P90'], [ttft['mean'] or 0, ttft['median'] or 0, ttft['p90'] or 0], color='steelblue')
    axes[0].set_title('Time To First Token (ms)')
    axes[0].set_ylabel('ms')
    
    # TPOT
    tpot = metrics['tpot_ms']
    axes[1].bar(['Mean', 'Median', 'P90'], [tpot['mean'] or 0, tpot['median'] or 0, tpot['p90'] or 0], color='coral')
    axes[1].set_title('Time Per Output Token (ms)')
    axes[1].set_ylabel('ms')
    
    # Throughput
    tp = metrics['throughput']
    axes[2].bar(['Req/s', 'Tok/s', 'Tok/s/user'], 
                [tp['requests_per_sec'] or 0, 
                 (tp['output_tokens_per_sec'] or 0) / 100,  # Scale for visibility
                 tp['tokens_per_sec_per_user'] or 0], 
                color='seagreen')
    axes[2].set_title('Throughput')
    axes[2].set_ylabel('Value (tok/s scaled by 100)')
    
    plt.tight_layout()
    plt.savefig("$plot_file", dpi=150)
    print(f"Plot saved to: $plot_file")
except Exception as e:
    print(f"Error generating plot: {e}")
EOF

    if [[ -f "$plot_file" ]]; then
        log_success "Timeline plot saved to: $plot_file"
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║      vLLM Online Benchmark Tool        ║${NC}"
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

    # Check server
    check_server

    # Run benchmark
    run_benchmark

    echo ""
    log_success "Benchmark complete!"
}

main "$@"
