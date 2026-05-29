export TC_PATH="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
export LD_PRELOAD="$TC_PATH:$VIRTUAL_ENV/lib/libiomp5.so"
export VLLM_CPU_SGL_KERNEL=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=32 
# export VLLM_CPU_OMP_THREADS_BIND="0-31|32-63"
export VLLM_CPU_OMP_THREADS_BIND="64-95|96-127"
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

# Debugging flags
# export TORCH_LOGS=+graph
# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_DISABLE_COMPILE_CACHE=1
# export  TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1