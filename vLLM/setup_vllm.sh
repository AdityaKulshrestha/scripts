#!/bin/bash

# =============================================================================
# vLLM Setup Script
# Supports: CPU, CUDA, XPU
# Package Managers: uv, conda
# Installation Methods: precompiled, build
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEVICE=""
METHOD=""
PKG_MANAGER=""
VENV_NAME=""
VLLM_VERSION=""

# =============================================================================
# Help Documentation
# =============================================================================
show_help() {
    cat << 'EOF'
================================================================================
                           vLLM Setup Script
================================================================================

DESCRIPTION:
    This script automates the installation of vLLM for different hardware
    backends (CPU, CUDA, XPU) using either precompiled packages or building
    from source.

USAGE:
    ./setup_vllm.sh [OPTIONS]

OPTIONS:
    -d, --device <device>       Target device: cpu, cuda, xpu
    -m, --method <method>       Installation method: precompiled, build
    -p, --pkg-manager <manager> Package manager: uv, conda
    -n, --name <name>           Virtual environment name
    -v, --version <version>     vLLM version (e.g., 0.23.0).
                                  - CPU: if not provided, uses the VLLM_CPU
                                    environment variable.
                                  - XPU: used to checkout the matching branch
                                    before building.
    -h, --help                  Show this help message

SUPPORTED PACKAGE MANAGERS:
    uv      - Fast Python package manager (recommended)
              Install: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    conda   - Anaconda/Miniconda package manager
              Install: https://docs.conda.io/en/latest/miniconda.html

SUPPORTED DEVICES:
    cpu     - CPU-only installation (Intel/AMD)
    cuda    - NVIDIA GPU with CUDA support
    xpu     - Intel GPU with XPU support (oneAPI)

INSTALLATION METHODS:
    precompiled  - Install pre-built wheels (faster, recommended)
    build        - Build from source (for customization/debugging)

EXAMPLES:
    # Install vLLM for CPU using precompiled package with uv
    ./setup_vllm.sh -d cpu -m precompiled -p uv

    # Build vLLM from source for CUDA
    ./setup_vllm.sh -d cuda -m build -p uv

    # Install for XPU (build only)
    ./setup_vllm.sh -d xpu -m build -p uv

    # Interactive mode (will prompt for options)
    ./setup_vllm.sh

NOTES:
    - XPU precompiled packages are not currently supported
    - Building from source requires additional system dependencies
    - If download fails, check your proxy settings

================================================================================
EOF
    exit 0
}

# =============================================================================
# Utility Functions
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_connectivity() {
    log_info "Checking internet connectivity..."
    if ! curl -s --connect-timeout 10 https://github.com > /dev/null 2>&1; then
        log_error "Cannot connect to the internet."
        log_error "Please check your proxy settings:"
        echo "  export http_proxy=http://your-proxy:port"
        echo "  export https_proxy=http://your-proxy:port"
        exit 1
    fi
    log_success "Internet connectivity OK"
}

# =============================================================================
# Package Manager Detection and Installation
# =============================================================================
select_pkg_manager() {
    if [[ -z "$PKG_MANAGER" ]]; then
        echo ""
        echo "Select package manager:"
        echo "  1) uv (recommended - fast and modern)"
        echo "  2) conda"
        read -p "Enter choice [1-2]: " choice
        case $choice in
            1) PKG_MANAGER="uv" ;;
            2) PKG_MANAGER="conda" ;;
            *) log_error "Invalid choice"; exit 1 ;;
        esac
    fi
}

check_pkg_manager() {
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        if command -v uv &> /dev/null; then
            log_success "uv is already installed: $(uv --version 2>/dev/null || echo 'unknown')"
        else
            log_warn "uv is not installed"
            install_uv
        fi
    elif [[ "$PKG_MANAGER" == "conda" ]]; then
        # Try multiple ways to detect conda
        if command -v conda &> /dev/null; then
            log_success "conda is already installed: $(conda --version 2>/dev/null || echo 'unknown')"
        elif [[ -n "$CONDA_EXE" ]]; then
            # Conda is installed but not in PATH, use CONDA_EXE
            log_success "conda found via CONDA_EXE: $CONDA_EXE"
            eval "$($CONDA_EXE shell.bash hook)"
        elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
            log_info "Found conda at ~/miniconda3, initializing..."
            eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
            log_success "conda initialized"
        elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
            log_info "Found conda at ~/anaconda3, initializing..."
            eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
            log_success "conda initialized"
        else
            log_error "conda is not installed or not found."
            log_info "Please install conda manually from:"
            echo "  https://docs.conda.io/en/latest/miniconda.html"
            echo ""
            log_info "Or initialize conda in your shell:"
            echo "  conda init bash"
            exit 1
        fi
    fi
}

install_uv() {
    log_info "Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$PATH"
        PKG_MANAGER="uv"
        log_success "uv installed successfully"
    else
        log_error "Failed to install uv."
        log_error "Please check your proxy settings and try again."
        exit 1
    fi
}

# =============================================================================
# Virtual Environment Setup
# =============================================================================
setup_venv_uv() {
    # Check if .venv already exists
    if [[ -d "$VENV_NAME" ]]; then
        log_info "Virtual environment '$VENV_NAME' already exists"
        read -p "Use existing environment? (y/n) [y]: " choice
        choice=${choice:-y}
        if [[ "${choice,,}" == "y" ]]; then
            source "$VENV_NAME/bin/activate"
            log_success "Activated existing environment: $VENV_NAME"
            return
        else
            log_info "Removing existing environment..."
            rm -rf "$VENV_NAME"
        fi
    fi
    
    log_info "Creating virtual environment with uv..."
    uv venv --python 3.12 --seed --managed-python "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    log_success "Virtual environment created and activated"
}

setup_venv_conda() {
    # Check if already in a conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        log_info "Currently in conda environment: $CONDA_DEFAULT_ENV"
        if [[ "$CONDA_DEFAULT_ENV" == "$VENV_NAME" ]]; then
            log_success "Already in target environment: $VENV_NAME"
            return
        else
            read -p "Use current environment '$CONDA_DEFAULT_ENV'? (y/n) [y]: " choice
            choice=${choice:-y}
            if [[ "${choice,,}" == "y" ]]; then
                VENV_NAME="$CONDA_DEFAULT_ENV"
                log_success "Using current environment: $VENV_NAME"
                return
            else
                # Ask for new environment name
                read -p "Enter conda environment name [vllm_env]: " new_env_name
                VENV_NAME="${new_env_name:-vllm_env}"
            fi
        fi
    fi
    
    # Check if target environment already exists
    if conda env list | grep -q "^$VENV_NAME "; then
        log_info "Conda environment '$VENV_NAME' already exists"
        read -p "Use existing environment? (y/n) [y]: " choice
        choice=${choice:-y}
        if [[ "${choice,,}" == "y" ]]; then
            eval "$(conda shell.bash hook)"
            conda activate "$VENV_NAME"
            log_success "Activated existing environment: $VENV_NAME"
            return
        else
            log_info "Removing existing environment..."
            conda env remove -n "$VENV_NAME" -y
        fi
    fi
    
    log_info "Creating conda environment '$VENV_NAME'..."
    conda create -n "$VENV_NAME" python=3.12 -y
    eval "$(conda shell.bash hook)"
    conda activate "$VENV_NAME"
    log_success "Conda environment created and activated"
}

setup_venv() {
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        setup_venv_uv
    else
        setup_venv_conda
    fi
}

# =============================================================================
# vLLM Source Checkout
# =============================================================================
# Clone the vLLM repository unless a source folder already exists.
# If 'vllm_source' or 'vllm' is already present, reuse it and cd into it.
clone_vllm() {
    if [[ -d "vllm_source" ]]; then
        log_info "vllm_source folder already exists, skipping clone."
        cd vllm_source
    elif [[ -d "vllm" ]]; then
        log_info "vllm folder already exists, skipping clone."
        cd vllm
    else
        log_info "Cloning vLLM repository..."
        git clone https://github.com/vllm-project/vllm.git vllm_source
        cd vllm_source
    fi
}

# =============================================================================
# CPU Installation
# =============================================================================
install_cpu_precompiled() {
    log_info "Installing vLLM for CPU (precompiled)..."
    
    setup_venv
    
    # Determine which vLLM version to install:
    #   - If the user provided a version, use it.
    #   - Otherwise, fall back to the VLLM_CPU environment variable.
    if [[ -z "$VLLM_VERSION" ]]; then
        if [[ -n "$VLLM_CPU" ]]; then
            VLLM_VERSION="$VLLM_CPU"
            log_info "No version provided, using VLLM_CPU env var: ${VLLM_VERSION}"
        else
            log_error "No vLLM version provided and VLLM_CPU env var is not set."
            log_error "Provide a version with -v/--version or export VLLM_CPU."
            exit 1
        fi
    else
        log_info "Using user-provided vLLM version: ${VLLM_VERSION}"
    fi
    
    # Normalize: strip a leading 'v' if present (e.g. v0.23.0 -> 0.23.0)
    VLLM_VERSION="${VLLM_VERSION#v}"
    
    log_info "Installing vLLM v${VLLM_VERSION}..."
    
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" --torch-backend cpu
    else
        pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"
    fi
    
    log_success "vLLM for CPU installed successfully!"
}

install_cpu_build() {
    log_info "Building vLLM for CPU from source..."
    
    log_info "Installing system dependencies..."
    sudo apt-get update -y
    sudo apt-get install -y gcc-12 g++-12 libnuma-dev
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
    
    setup_venv
    
    clone_vllm
    
    log_info "Installing build dependencies..."
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        uv pip install -r requirements/build/cpu.txt --torch-backend cpu
        uv pip install -r requirements/cpu.txt --torch-backend cpu
    else
        pip install -r requirements/build/cpu.txt
        pip install -r requirements/cpu.txt
    fi
    
    log_info "Building vLLM..."
    VLLM_TARGET_DEVICE=cpu python3 setup.py develop
    
    log_success "vLLM for CPU built successfully!"
}

# =============================================================================
# CUDA Installation
# =============================================================================
install_cuda_precompiled() {
    log_info "Installing vLLM for CUDA (precompiled)..."
    
    setup_venv
    
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        uv pip install vllm
    else
        pip install vllm
    fi
    
    log_success "vLLM for CUDA installed successfully!"
}

install_cuda_build() {
    log_info "Building vLLM for CUDA from source..."
    
    setup_venv
    
    clone_vllm
    
    log_info "Building vLLM..."
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        uv pip install -e . --torch-backend=auto
    else
        pip install -e .
    fi
    
    log_success "vLLM for CUDA built successfully!"
}

# =============================================================================
# XPU Installation
# =============================================================================
install_xpu_precompiled() {
    log_error "Precompiled packages for XPU are not currently supported."
    log_info "Please use the build method instead:"
    echo "  ./setup_vllm.sh -d xpu -m build"
    exit 1
}

install_xpu_build() {
    log_info "Building vLLM for XPU from source..."
    
    setup_venv
    
    clone_vllm
    
    if [[ -n "$VLLM_VERSION" ]]; then
        # Normalize: strip a leading 'v' if present (e.g. v0.23.0 -> 0.23.0)
        VLLM_VERSION="${VLLM_VERSION#v}"
        log_info "Checking out vLLM branch v${VLLM_VERSION}..."
        git checkout "v${VLLM_VERSION}"
    else
        log_warn "No vLLM version provided, using the default branch."
    fi
    
    log_info "Installing dependencies..."
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        uv pip install --upgrade pip
        uv pip install -v -r requirements/xpu.txt --index-strategy unsafe-best-match
        
        log_info "Building vLLM..."
        VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation -e . -v
        
        log_info "Installing triton-xpu..."
        uv pip uninstall -y triton triton-xpu 2>/dev/null || true
        uv pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
    else
        pip install --upgrade pip
        pip install -v -r requirements/xpu.txt
        
        log_info "Building vLLM..."
        VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
        
        log_info "Installing triton-xpu..."
        pip uninstall -y triton triton-xpu 2>/dev/null || true
        pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
    fi
    
    log_success "vLLM for XPU built successfully!"
}

# =============================================================================
# Interactive Selection
# =============================================================================
select_device() {
    if [[ -z "$DEVICE" ]]; then
        echo ""
        echo "Select target device:"
        echo "  1) cpu  - CPU only"
        echo "  2) cuda - NVIDIA GPU"
        echo "  3) xpu  - Intel GPU"
        read -p "Enter choice [1-3]: " choice
        case $choice in
            1) DEVICE="cpu" ;;
            2) DEVICE="cuda" ;;
            3) DEVICE="xpu" ;;
            *) log_error "Invalid choice"; exit 1 ;;
        esac
    fi
}

select_method() {
    if [[ -z "$METHOD" ]]; then
        echo ""
        echo "Select installation method:"
        echo "  1) precompiled - Install pre-built packages (faster)"
        echo "  2) build       - Build from source"
        read -p "Enter choice [1-2]: " choice
        case $choice in
            1) METHOD="precompiled" ;;
            2) METHOD="build" ;;
            *) log_error "Invalid choice"; exit 1 ;;
        esac
    fi
}

select_venv_name() {
    if [[ -z "$VENV_NAME" ]]; then
        echo ""
        read -p "Enter virtual environment name [.vllm]: " input_name
        VENV_NAME="${input_name:-.vllm}"
    fi
}

select_version() {
    if [[ -z "$VLLM_VERSION" ]]; then
        echo ""
        read -p "Enter vLLM version (leave empty to use default/VLLM_CPU env): " input_version
        VLLM_VERSION="$input_version"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                ;;
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -m|--method)
                METHOD="$2"
                shift 2
                ;;
            -p|--pkg-manager)
                PKG_MANAGER="$2"
                shift 2
                ;;
            -n|--name)
                VENV_NAME="$2"
                shift 2
                ;;
            -v|--version)
                VLLM_VERSION="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=========================================="
    echo "         vLLM Setup Script"
    echo "=========================================="
    echo ""
    
    parse_args "$@"
    
    # Check connectivity first
    check_connectivity
    
    # Select and check package manager
    select_pkg_manager
    check_pkg_manager
    
    # Interactive selection if not provided
    select_device
    select_method
    select_venv_name
    select_version
    
    echo ""
    log_info "Configuration:"
    echo "  Device:          $DEVICE"
    echo "  Method:          $METHOD"
    echo "  Package Manager: $PKG_MANAGER"
    echo "  Environment:     $VENV_NAME"
    echo ""
    
    # Execute installation
    case "$DEVICE" in
        cpu)
            if [[ "$METHOD" == "precompiled" ]]; then
                install_cpu_precompiled
            else
                install_cpu_build
            fi
            ;;
        cuda)
            if [[ "$METHOD" == "precompiled" ]]; then
                install_cuda_precompiled
            else
                install_cuda_build
            fi
            ;;
        xpu)
            if [[ "$METHOD" == "precompiled" ]]; then
                install_xpu_precompiled
            else
                install_xpu_build
            fi
            ;;
        *)
            log_error "Invalid device: $DEVICE"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Installation complete!"
    echo ""
    echo "To activate the environment:"
    if [[ "$PKG_MANAGER" == "uv" ]]; then
        echo "  source $VENV_NAME/bin/activate"
    else
        echo "  conda activate $VENV_NAME"
    fi
    echo ""
}

main "$@"
