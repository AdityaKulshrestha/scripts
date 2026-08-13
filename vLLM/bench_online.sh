#!/bin/bash

# =============================================================================
# vLLM Online Benchmark Script
# Supports: Interactive & Command-line modes, and parameter sweeps
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
TASK="llm"           # llm | embed | reranker
MODEL=""
INPUT_TOKENS=""      # can be a comma-separated list, e.g. "1024,512,128"
OUTPUT_TOKENS=""     # can be a comma-separated list, e.g. "1024,512"
CONCURRENCY=""       # can be a comma-separated list (batch size), e.g. "8,16"
NUM_PROMPTS=""
REQUEST_RATE="inf"
DATASET="random"
TOKENIZER=""
RESULTS_DIR="./results"
RESULT_FILENAME=""
APPEND_RESULT=false
INTERACTIVE=false

# Sweep summary CSV (appended to after every single benchmark run)
SWEEP_CSV=""

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
                    vLLM Online Benchmark Tool
================================================================================

USAGE:
    ./run_benchmark.sh [OPTIONS]
    ./run_benchmark.sh --interactive

MODES:
    Command-line    Provide all options via arguments
    Interactive     Prompt for options (use --interactive or -i)
    Sweep           Pass comma-separated lists to --input-tokens,
                    --output-tokens and/or --concurrency to benchmark
                    every combination automatically.

TASKS:
    llm         Chat/completions benchmarking
    embed       Embeddings benchmarking
    reranker    Reranker benchmarking

REQUIRED OPTIONS (command-line mode):
    --task <llm|embed|reranker>      Benchmark task (default: llm)
    --model <name>                  Model name/path
    --input-tokens <num[,num...]>   Input token length(s), e.g. "1024,512,128"
    --output-tokens <num[,num...]>  Output token length(s), required for llm only
    --concurrency <num[,num...]>    Max concurrency (llm/embed) or random batch size (reranker)
    --num-prompts <num>             Total number of prompts (applied to every combo)

OPTIONAL OPTIONS:
    --base-url <url>            Server URL (default: http://localhost:8000)
    --request-rate <rate>       Request rate (default: inf)
    --dataset <name>            Dataset: random, sharegpt, sonnet (default: random)
    --tokenizer <name>          Tokenizer name/path (optional; reranker defaults to model)
    --results-dir <path>        Results directory (default: ./results)
    --result-filename <name>    Custom result filename prefix (without extension)
    --append-result             Append to existing per-run result file
    --sweep-csv <path>          Sweep summary CSV path
                                 (default: <results-dir>/sweep_results.csv)
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
    --plot-timeline             Generate a timeline plot after each benchmark run

PASS-THROUGH:
    --bench-args <args...>      Additional args for vllm bench serve (must be last)

SWEEP BEHAVIOR:
    When any of --input-tokens / --output-tokens / --concurrency contains more
    than one comma-separated value, the script runs every combination
    (cartesian product) back to back. After each individual run finishes
    (success OR failure), a row is immediately appended to the sweep summary
    CSV so results are never lost if a later run crashes or is interrupted.
    Each run also still writes its own per-run JSON + log file, named with
    its input/output/concurrency values so they never collide.

EXAMPLES:
    # Single command-line run
    ./run_benchmark.sh --model meta-llama/Llama-2-7b \
        --input-tokens 128 --output-tokens 128 \
        --concurrency 4 --num-prompts 16

    # Embedding benchmark
    ./run_benchmark.sh --task embed --model BAAI/bge-large-en-v1.5 \
        --input-tokens 256 --concurrency 32 --num-prompts 128

    # Reranker benchmark
    ./run_benchmark.sh --task reranker --model BAAI/bge-reranker-v2-m3 \
        --tokenizer BAAI/bge-reranker-v2-m3 \
        --input-tokens 512 --concurrency 5 --num-prompts 10

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

    # Full parameter sweep: 3 input lens x 2 output lens x 2 batch sizes = 12 runs,
    # results appended to results/sweep_results.csv as each run completes
    ./run_benchmark.sh --model meta-llama/Llama-2-7b \
        --input-tokens 1024,512,128 \
        --output-tokens 1024,512 \
        --concurrency 8,16 \
        --num-prompts 64 \
        --results-dir ./results

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

    # Task
    echo "Task options: llm, embed, reranker"
    read -p "Task [llm]: " input
    TASK="${input:-llm}"

    case "$TASK" in
        llm|embed|reranker) ;;
        *) log_error "Invalid task: $TASK (must be llm, embed, or reranker)"; exit 1 ;;
    esac

    # Input tokens
    read -p "Input tokens, comma-separated for a sweep e.g. 1024,512,128 (required): " INPUT_TOKENS
    [[ -z "$INPUT_TOKENS" ]] && { log_error "Input tokens required"; exit 1; }

    # Output tokens (LLM-only)
    if [[ "$TASK" == "llm" ]]; then
        read -p "Output tokens, comma-separated for a sweep e.g. 1024,512 (required for llm): " OUTPUT_TOKENS
        [[ -z "$OUTPUT_TOKENS" ]] && { log_error "Output tokens required for llm"; exit 1; }
    else
        OUTPUT_TOKENS="0"
    fi

    # Concurrency / rerank random-batch-size
    if [[ "$TASK" == "reranker" ]]; then
        read -p "Reranker random batch size(s), comma-separated e.g. 5,10 (required): " CONCURRENCY
    else
        read -p "Concurrency / batch size(s), comma-separated e.g. 8,16 (required): " CONCURRENCY
    fi
    [[ -z "$CONCURRENCY" ]] && { log_error "Concurrency required"; exit 1; }

    # Num prompts
    read -p "Number of prompts per run (required): " NUM_PROMPTS
    [[ -z "$NUM_PROMPTS" ]] && { log_error "Num prompts required"; exit 1; }

    # Request rate
    read -p "Request rate [inf]: " input
    REQUEST_RATE="${input:-inf}"

    # Dataset
    if [[ "$TASK" == "llm" ]]; then
        echo "Dataset options: random, sharegpt, sonnet"
        read -p "Dataset [random]: " input
        DATASET="${input:-random}"
    elif [[ "$TASK" == "embed" ]]; then
        DATASET="random"
        log_info "Task=embed -> dataset fixed to random"
    else
        DATASET="random-rerank"
        log_info "Task=reranker -> dataset fixed to random-rerank"

        read -p "Tokenizer [${MODEL}]: " input
        TOKENIZER="${input:-$MODEL}"
    fi

    # Results directory
    read -p "Results directory [./results]: " input
    RESULTS_DIR="${input:-./results}"

    # Sweep CSV
    read -p "Sweep summary CSV [${RESULTS_DIR}/sweep_results.csv]: " input
    SWEEP_CSV="${input:-${RESULTS_DIR}/sweep_results.csv}"

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
    read -p "Generate timeline plot per run? (y/n) [n]: " input
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
            --task) TASK="$2"; shift 2 ;;
            --model) MODEL="$2"; shift 2 ;;
            --input-tokens) INPUT_TOKENS="$2"; shift 2 ;;
            --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
            --concurrency) CONCURRENCY="$2"; shift 2 ;;
            --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
            --request-rate) REQUEST_RATE="$2"; shift 2 ;;
            --dataset) DATASET="$2"; shift 2 ;;
            --tokenizer) TOKENIZER="$2"; shift 2 ;;
            --results-dir) RESULTS_DIR="$2"; shift 2 ;;
            --result-filename) RESULT_FILENAME="$2"; shift 2 ;;
            --append-result) APPEND_RESULT=true; shift ;;
            --sweep-csv) SWEEP_CSV="$2"; shift 2 ;;
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

    case "$TASK" in
        llm|embed|reranker) ;;
        *)
            log_error "Invalid --task '$TASK'. Must be one of: llm, embed, reranker"
            exit 1
            ;;
    esac

    [[ -z "$MODEL" ]] && missing+=("--model")
    [[ -z "$INPUT_TOKENS" ]] && missing+=("--input-tokens")
    if [[ "$TASK" == "llm" && -z "$OUTPUT_TOKENS" ]]; then
        missing+=("--output-tokens")
    fi
    [[ -z "$CONCURRENCY" ]] && missing+=("--concurrency")
    [[ -z "$NUM_PROMPTS" ]] && missing+=("--num-prompts")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required arguments:"
        printf '  %s\n' "${missing[@]}"
        echo ""
        echo "Use --help for usage or -i for interactive mode"
        exit 1
    fi

    if [[ "$TASK" == "embed" ]]; then
        DATASET="random"
        OUTPUT_TOKENS="0"
    elif [[ "$TASK" == "reranker" ]]; then
        DATASET="random-rerank"
        OUTPUT_TOKENS="0"
        [[ -z "$TOKENIZER" ]] && TOKENIZER="$MODEL"
    fi

    if [[ -z "$SWEEP_CSV" ]]; then
        SWEEP_CSV="${RESULTS_DIR}/sweep_results.csv"
    fi
}

# =============================================================================
# Build Benchmark Command
# Uses the CURRENT (per-iteration) scalar values of INPUT_TOKENS, OUTPUT_TOKENS
# and CONCURRENCY - these are set by run_sweep() before each individual run.
# =============================================================================
build_bench_cmd() {
    local cmd="vllm bench serve"
    cmd+=" --base-url $BASE_URL"
    cmd+=" --model $MODEL"
    cmd+=" --metric-percentiles 90"

    case "$TASK" in
        llm)
            cmd+=" --backend openai-chat"
            cmd+=" --endpoint /v1/chat/completions"
            cmd+=" --ignore-eos"

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
            ;;
        embed)
            cmd+=" --backend openai-embeddings"
            cmd+=" --endpoint /v1/embeddings"
            cmd+=" --dataset-name random"
            cmd+=" --random-input-len $INPUT_TOKENS"
            cmd+=" --request-rate $REQUEST_RATE"
            cmd+=" --num-prompts $NUM_PROMPTS"
            cmd+=" --max-concurrency $CONCURRENCY"
            ;;
        reranker)
            cmd+=" --backend vllm-rerank"
            cmd+=" --endpoint /v1/rerank"
            cmd+=" --dataset-name random-rerank"
            cmd+=" --tokenizer $TOKENIZER"
            cmd+=" --random-input-len $INPUT_TOKENS"
            cmd+=" --num-prompts $NUM_PROMPTS"
            cmd+=" --random-batch-size $CONCURRENCY"
            ;;
    esac

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
# Extract Metrics from raw benchmark output.
# Populates the ME_* globals used by both the per-run JSON and the sweep CSV.
# =============================================================================
extract_metrics() {
    local output="$1"

    ME_MEAN_TTFT=$(echo "$output" | grep -E "^Mean TTFT \(ms\):" | awk '{print $4}' || echo "")
    ME_MEDIAN_TTFT=$(echo "$output" | grep -E "^Median TTFT \(ms\):" | awk '{print $4}' || echo "")
    ME_P90_TTFT=$(echo "$output" | grep -E "^P90 TTFT \(ms\):" | awk '{print $4}' || echo "")

    ME_MEAN_TPOT=$(echo "$output" | grep -E "^Mean TPOT \(ms\):" | awk '{print $4}' || echo "")
    ME_MEDIAN_TPOT=$(echo "$output" | grep -E "^Median TPOT \(ms\):" | awk '{print $4}' || echo "")
    ME_P90_TPOT=$(echo "$output" | grep -E "^P90 TPOT \(ms\):" | awk '{print $4}' || echo "")

    ME_MEAN_ITL=$(echo "$output" | grep -E "^Mean ITL \(ms\):" | awk '{print $4}' || echo "")
    ME_MEDIAN_ITL=$(echo "$output" | grep -E "^Median ITL \(ms\):" | awk '{print $4}' || echo "")
    ME_P90_ITL=$(echo "$output" | grep -E "^P90 ITL \(ms\):" | awk '{print $4}' || echo "")

    ME_REQ_THROUGHPUT=$(echo "$output" | grep -E "^Request throughput \(req/s\):" | awk '{print $4}' || echo "")
    ME_OUTPUT_THROUGHPUT=$(echo "$output" | grep -E "^Output token throughput \(tok/s\):" | awk '{print $5}' || echo "")

    ME_INTERACTIVITY=""
    if [[ -n "$ME_OUTPUT_THROUGHPUT" && "$CONCURRENCY" -gt 0 ]]; then
        ME_INTERACTIVITY=$(echo "scale=2; $ME_OUTPUT_THROUGHPUT / $CONCURRENCY" | bc 2>/dev/null || echo "")
    fi
}

# =============================================================================
# Save Results as JSON (single run)
# =============================================================================
save_results_json() {
    local output="$1"

    extract_metrics "$output"

    # Build JSON
    local json_result
    json_result=$(cat <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "config": {
        "task": "$TASK",
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
      "mean": ${ME_MEAN_TTFT:-null},
      "median": ${ME_MEDIAN_TTFT:-null},
      "p90": ${ME_P90_TTFT:-null}
    },
    "tpot_ms": {
      "mean": ${ME_MEAN_TPOT:-null},
      "median": ${ME_MEDIAN_TPOT:-null},
      "p90": ${ME_P90_TPOT:-null}
    },
    "itl_ms": {
      "mean": ${ME_MEAN_ITL:-null},
      "median": ${ME_MEDIAN_ITL:-null},
      "p90": ${ME_P90_ITL:-null}
    },
    "throughput": {
      "requests_per_sec": ${ME_REQ_THROUGHPUT:-null},
      "output_tokens_per_sec": ${ME_OUTPUT_THROUGHPUT:-null},
      "tokens_per_sec_per_user": ${ME_INTERACTIVITY:-null}
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
        local existing
        existing=$(cat "$RESULT_FILE")
        if echo "$existing" | grep -q '^\['; then
            existing="${existing%]}"
            echo "${existing},${json_result}]" > "$RESULT_FILE"
        else
            echo "[${existing},${json_result}]" > "$RESULT_FILE"
        fi
    else
        echo "$json_result" > "$RESULT_FILE"
    fi
}

# =============================================================================
# Append one row to the sweep summary CSV.
# Writes the header first if the file doesn't exist yet.
# Called after EVERY run (success or failure) so progress is never lost.
# =============================================================================
init_sweep_csv() {
    if [[ ! -f "$SWEEP_CSV" ]]; then
        mkdir -p "$(dirname "$SWEEP_CSV")"
        echo "timestamp,status,task,model,dataset,input_tokens,output_tokens,concurrency,num_prompts,request_rate,mean_ttft_ms,median_ttft_ms,p90_ttft_ms,mean_tpot_ms,median_tpot_ms,p90_tpot_ms,mean_itl_ms,median_itl_ms,p90_itl_ms,req_throughput_per_s,output_tok_throughput_per_s,tok_per_s_per_user,result_file,log_file" > "$SWEEP_CSV"
    fi
}

append_sweep_row() {
    local status="$1"

    local row
    row="$(date -Iseconds),${status},${TASK},${MODEL},${DATASET},${INPUT_TOKENS},${OUTPUT_TOKENS},${CONCURRENCY},${NUM_PROMPTS},${REQUEST_RATE},${ME_MEAN_TTFT:-},${ME_MEDIAN_TTFT:-},${ME_P90_TTFT:-},${ME_MEAN_TPOT:-},${ME_MEDIAN_TPOT:-},${ME_P90_TPOT:-},${ME_MEAN_ITL:-},${ME_MEDIAN_ITL:-},${ME_P90_ITL:-},${ME_REQ_THROUGHPUT:-},${ME_OUTPUT_THROUGHPUT:-},${ME_INTERACTIVITY:-},${RESULT_FILE:-},${LOG_FILE:-}"

    echo "$row" >> "$SWEEP_CSV"
}

# =============================================================================
# Generate Timeline Plot (single run)
# =============================================================================
generate_timeline_plot() {
    log_info "Generating timeline plot..."

    if ! python3 -c "import matplotlib" 2>/dev/null; then
        log_warn "matplotlib not installed. Skipping plot generation."
        echo "  Install with: pip install matplotlib"
        return
    fi

    local plot_file="${RESULTS_DIR}/timeline_$(date +%Y%m%d_%H%M%S)_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_c${CONCURRENCY}.png"

    RESULT_FILE="$RESULT_FILE" PLOT_FILE="$plot_file" python3 << 'EOF'
import json
import os
import matplotlib.pyplot as plt

result_file = os.environ["RESULT_FILE"]
plot_file = os.environ["PLOT_FILE"]

try:
    with open(result_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        data = data[-1]

    metrics = data['metrics']
    config = data['config']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Benchmark: {config['model']} | in={config['input_tokens']} "
        f"out={config['output_tokens']} conc={config['concurrency']}",
        fontsize=13,
    )

    ttft = metrics['ttft_ms']
    axes[0].bar(['Mean', 'Median', 'P90'], [ttft['mean'] or 0, ttft['median'] or 0, ttft['p90'] or 0], color='steelblue')
    axes[0].set_title('Time To First Token (ms)')
    axes[0].set_ylabel('ms')

    tpot = metrics['tpot_ms']
    axes[1].bar(['Mean', 'Median', 'P90'], [tpot['mean'] or 0, tpot['median'] or 0, tpot['p90'] or 0], color='coral')
    axes[1].set_title('Time Per Output Token (ms)')
    axes[1].set_ylabel('ms')

    tp = metrics['throughput']
    axes[2].bar(['Req/s', 'Tok/s', 'Tok/s/user'],
                [tp['requests_per_sec'] or 0,
                 (tp['output_tokens_per_sec'] or 0) / 100,
                 tp['tokens_per_sec_per_user'] or 0],
                color='seagreen')
    axes[2].set_title('Throughput')
    axes[2].set_ylabel('Value (tok/s scaled by 100)')

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to: {plot_file}")
except Exception as e:
    print(f"Error generating plot: {e}")
EOF

    if [[ -f "$plot_file" ]]; then
        log_success "Timeline plot saved to: $plot_file"
    fi
}

# =============================================================================
# Run ONE benchmark for the CURRENT scalar INPUT_TOKENS / OUTPUT_TOKENS /
# CONCURRENCY values. Never exits the process on failure - returns non-zero
# so the sweep can continue to the next combination.
# =============================================================================
run_single_benchmark() {
    mkdir -p "$RESULTS_DIR"
    $PROFILE && mkdir -p "${PROFILE_DIR}/in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_c${CONCURRENCY}"

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local model_short
    model_short=$(basename "$MODEL" | tr '/' '_')

    local combo_tag="task${TASK}_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_c${CONCURRENCY}"
    local prefix="${RESULT_FILENAME:-benchmark_${TASK}_${model_short}}"

    RESULT_FILE="${RESULTS_DIR}/${prefix}_${combo_tag}_${timestamp}.json"
    LOG_FILE="${RESULTS_DIR}/${prefix}_${combo_tag}_${timestamp}.log"

    if $PROFILE; then
        # keep per-combo profiling isolated
        RUN_PROFILE_DIR="${PROFILE_DIR}/${combo_tag}"
    fi

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}        Benchmark Configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "  Server:       $BASE_URL"
    echo "  Task:         $TASK"
    echo "  Model:        $MODEL"
    echo "  Input:        $INPUT_TOKENS tokens"
    echo "  Output:       $OUTPUT_TOKENS tokens"
    if [[ "$TASK" == "reranker" ]]; then
        echo "  Rerank Batch: $CONCURRENCY"
        echo "  Tokenizer:    $TOKENIZER"
    else
        echo "  Concurrency:  $CONCURRENCY"
    fi
    echo "  Prompts:      $NUM_PROMPTS"
    echo "  Request Rate: $REQUEST_RATE"
    echo "  Dataset:      $DATASET"
    echo "  Results:      $RESULT_FILE"
    $PROFILE && echo "  Profile Dir:  $RUN_PROFILE_DIR"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    local cmd
    if $PROFILE; then
        local saved_profile_dir="$PROFILE_DIR"
        PROFILE_DIR="$RUN_PROFILE_DIR"
        cmd=$(build_bench_cmd)
        PROFILE_DIR="$saved_profile_dir"
    else
        cmd=$(build_bench_cmd)
    fi

    log_info "Running benchmark..."
    echo "Command: $cmd"
    echo ""

    local output
    local exit_code=0
    output=$(eval "$cmd" 2>&1) || exit_code=$?

    echo "$output" > "$LOG_FILE"

    if ! echo "$output" | grep -q "Serving Benchmark Result"; then
        log_error "Benchmark failed for input=${INPUT_TOKENS} output=${OUTPUT_TOKENS} concurrency=${CONCURRENCY}"
        echo "$output" | tail -20
        # clear metrics so the CSV row records blanks, and log a failure row
        ME_MEAN_TTFT="" ME_MEDIAN_TTFT="" ME_P90_TTFT=""
        ME_MEAN_TPOT="" ME_MEDIAN_TPOT="" ME_P90_TPOT=""
        ME_MEAN_ITL="" ME_MEDIAN_ITL="" ME_P90_ITL=""
        ME_REQ_THROUGHPUT="" ME_OUTPUT_THROUGHPUT="" ME_INTERACTIVITY=""
        append_sweep_row "FAILED"
        return 1
    fi

    log_success "Benchmark completed"
    echo ""

    save_results_json "$output"

    echo ""
    echo -e "${CYAN}Key Metrics:${NC}"
    echo "$output" | grep -E "(Mean TTFT|Mean TPOT|Mean ITL|Request throughput|Output token throughput)" | head -10
    echo ""

    if $PLOT_TIMELINE; then
        generate_timeline_plot
    fi

    append_sweep_row "SUCCESS"

    echo ""
    log_success "Results saved to: $RESULT_FILE"
    echo "Log saved to: $LOG_FILE"
    echo "Sweep summary appended to: $SWEEP_CSV"

    return 0
}

# =============================================================================
# Run the full sweep: cartesian product of INPUT_TOKENS x OUTPUT_TOKENS x
# CONCURRENCY (each of which may be a single value or a comma-separated list).
# =============================================================================
run_sweep() {
    local input_list_str="$INPUT_TOKENS"
    local output_list_str="$OUTPUT_TOKENS"
    local conc_list_str="$CONCURRENCY"

    local -a input_arr output_arr conc_arr
    IFS=',' read -ra input_arr <<< "$input_list_str"
    IFS=',' read -ra output_arr <<< "$output_list_str"
    IFS=',' read -ra conc_arr <<< "$conc_list_str"

    # trim whitespace on each element
    local i
    for i in "${!input_arr[@]}"; do input_arr[$i]=$(echo "${input_arr[$i]}" | xargs); done
    for i in "${!output_arr[@]}"; do output_arr[$i]=$(echo "${output_arr[$i]}" | xargs); done
    for i in "${!conc_arr[@]}"; do conc_arr[$i]=$(echo "${conc_arr[$i]}" | xargs); done

    local total=$(( ${#input_arr[@]} * ${#output_arr[@]} * ${#conc_arr[@]} ))

    init_sweep_csv

    if [[ $total -gt 1 ]]; then
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}   Sweep Mode: ${total} combinations${NC}"
        echo -e "${CYAN}   input-tokens:  ${input_list_str}${NC}"
        echo -e "${CYAN}   output-tokens: ${output_list_str}${NC}"
        echo -e "${CYAN}   concurrency:   ${conc_list_str}${NC}"
        echo -e "${CYAN}   sweep CSV:     ${SWEEP_CSV}${NC}"
        echo -e "${CYAN}========================================${NC}"
    fi

    local run_idx=0
    local fail_count=0
    local i_val o_val c_val

    for i_val in "${input_arr[@]}"; do
        for o_val in "${output_arr[@]}"; do
            for c_val in "${conc_arr[@]}"; do
                run_idx=$((run_idx + 1))
                if [[ $total -gt 1 ]]; then
                    echo ""
                    log_info "[$run_idx/$total] input=${i_val} output=${o_val} concurrency=${c_val}"
                fi

                # set the per-iteration scalar values used everywhere else
                INPUT_TOKENS="$i_val"
                OUTPUT_TOKENS="$o_val"
                CONCURRENCY="$c_val"

                if ! run_single_benchmark; then
                    fail_count=$((fail_count + 1))
                fi
            done
        done
    done

    # restore list strings in case caller needs them
    INPUT_TOKENS="$input_list_str"
    OUTPUT_TOKENS="$output_list_str"
    CONCURRENCY="$conc_list_str"

    echo ""
    if [[ $total -gt 1 ]]; then
        echo -e "${CYAN}========================================${NC}"
        log_success "Sweep complete: $((total - fail_count))/${total} runs succeeded"
        [[ $fail_count -gt 0 ]] && log_warn "${fail_count} run(s) failed - see sweep CSV for details"
        echo "Sweep summary CSV: $SWEEP_CSV"
        echo -e "${CYAN}========================================${NC}"
    fi

    if [[ $fail_count -eq $total ]]; then
        exit 1
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

    # Run benchmark(s) - handles both single-run and sweep cases
    run_sweep

    echo ""
    log_success "Benchmark run(s) complete!"
}

main "$@"
