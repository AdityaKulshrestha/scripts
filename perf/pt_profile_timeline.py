import ast
import sys
import json
import pandas as pd
import argparse
from collections import defaultdict
import time

# Argument parser setup for command-line options
parser = argparse.ArgumentParser(description="Process profiler JSON timeline.")
parser.add_argument("json_file", nargs="?", help="Path to the JSON file")
parser.add_argument("--summary-only", "-s", action="store_true", help="Print only the summary, not per-event details")
parser.add_argument("--ignore-shape", "-i", action="store_true", help="Ignore input shape in the summary key")
parser.add_argument("--no-header", "-H", action="store_true", help="Do not print the header for the output")
parser.add_argument("--max-depth", "-d", type=int, default=0, help="Maximum depth for nested events (default: 0 / unlimited)")
parser.add_argument("--nested-gpu-time", "-n", action="store_true", help="Calculate nested GPU execution time for events")
parser.add_argument("--kernel-map", "-k", action="store_true", help="Print GPU kernel map")
parser.add_argument("--kernel-only", "-K", action="store_true", help="Process only kernel events (exclude cpu_op, gpu_memcpy, etc.)")
parser.add_argument("--python-stack", "-p", action="store_true", help="Include Python Stack information if available")
parser.add_argument("--device-runtime", "-r", action="store_true", help="Include Device runtime calls")
parser.add_argument("--debug-time", "-t", action="store_true", help="Print debug timing information")
parser.add_argument("--output-excel-path", "-e", default=None, help="Output summary to an Excel file", type=str)
parser.add_argument("--peak-tflops", type=float, default=10.0, help="Peak TFLOPS of the system for utilization calculation (default: 10.0)")
parser.add_argument("--peak-bandwidth", type=float, default=400.0, help="Peak memory bandwidth in GB/s for utilization calculation (default: 400.0)")


# Parse arguments
args = parser.parse_args()

# Check for required JSON file argument
if not args.json_file:
    print(f"Usage: {sys.argv[0]} <json_file> [--summary-only] [--with-shape]")
    sys.exit(1)

# Extract argument values
json_file_path = args.json_file
summary_only = args.summary_only
with_shape = not args.ignore_shape
print_header = not args.no_header
is_gzipped = json_file_path.endswith(".gz")
max_depth = args.max_depth
nested_gpu_time = args.nested_gpu_time
print_kernel_map = args.kernel_map
kernel_only = args.kernel_only
include_python_stack = args.python_stack
include_device_runtime = args.device_runtime

start_load_time = time.time()
# Print timing information only if debug-time flag is set
def debug_time_print(*msg):
    if args.debug_time:
        print(*msg, file=sys.stderr, flush=True)

# Load profiler data from JSON (supports gzip)
try:
    if is_gzipped:
        import gzip
        with gzip.open(json_file_path, 'rt', encoding='utf-8') as file:
            profiler_data = json.load(file)
    else:
        with open(json_file_path, 'r', encoding='utf-8') as file:
        #with open(json_file_path, 'r', encoding='latin1') as file:
            profiler_data = json.load(file)
except Exception as error:
    print(f"Error loading JSON file: {error}")
    sys.exit(1)

t0 = time.time()
# Print loading time
debug_time_print(f"Loaded JSON data from {json_file_path} in {t0 - start_load_time:.2f} seconds.")
# Extract trace events from loaded data
trace_events = profiler_data.get("traceEvents", [])
if not trace_events:
    print("No trace events found in the JSON data.")
    sys.exit(0)
t1 = time.time()
# Print loading time
debug_time_print(f"Loaded {len(trace_events)} trace events in {t1 - t0:.2f} seconds.")
# Dictionaries to organize CPU and GPU events
cpu_events_by_thread = {}
gpu_events_by_external_id = {}
all_cpu_events = []
skipped_names = ["torch/nn/modules/module.py", "torch/autograd/function.py", "built-in function", "built-in method", "<module>", "__call__", ": forward"]
categories_to_process = ["cpu_op", "gpu_op"]
if include_device_runtime:
    categories_to_process.append("xpu_runtime")
    categories_to_process.append("cuda_runtime")
if include_python_stack:
    categories_to_process.append("python_function")

# Categorize events into CPU and GPU, and group them
for event in trace_events:
    if "ph" in event and event["ph"] == "X":
        if "pid" in event and "tid" in event:
            process_thread_id = (event["pid"], event["tid"])
            category = event.get("cat", "N/A")
            name = event.get("name", "N/A")
            skip = False
            for s in skipped_names:
                if s in name:
                    skip = True
            if skip: continue

            # Kernel-only mode: process only kernel events
            if kernel_only:
                if category == "kernel":
                    # Add to CPU events list for processing
                    if process_thread_id not in cpu_events_by_thread:
                        cpu_events_by_thread[process_thread_id] = []
                    cpu_events_by_thread[process_thread_id].append(event)
                    all_cpu_events.append(event)

                    # Also add to GPU events if it has External id (for parent linking)
                    if "args" in event and "External id" in event["args"]:
                        external_id = event["args"]["External id"]
                        if external_id not in gpu_events_by_external_id:
                            gpu_events_by_external_id[external_id] = []
                        gpu_events_by_external_id[external_id].append(event)
                # Skip all other categories in kernel-only mode
                continue

            # Original logic for non-kernel-only mode
            if category in categories_to_process:
                if process_thread_id not in cpu_events_by_thread:
                    cpu_events_by_thread[process_thread_id] = []
                cpu_events_by_thread[process_thread_id].append(event)
                all_cpu_events.append(event)
            elif "args" in event and "External id" in event["args"] and "device" in event["args"]:
                external_id = event["args"]["External id"]
                if external_id not in gpu_events_by_external_id:
                    gpu_events_by_external_id[external_id] = []
                gpu_events_by_external_id[external_id].append(event)
t2 = time.time()
# Print categorization time
debug_time_print(f"Categorized events in {t2 - t1:.2f} seconds. Found {len(all_cpu_events)} CPU events and {len(gpu_events_by_external_id)} GPU events.")
# Sort GPU events by timestamp for each external id
for external_id, events in gpu_events_by_external_id.items():
    events.sort(key=lambda x: x["ts"])
t3 = time.time()
# Print sorting time
debug_time_print(f"Sorted GPU events in {t3 - t2:.2f} seconds.")
has_gpu_events = len(gpu_events_by_external_id) > 0

# Sort all CPU events by start time and duration
all_cpu_events.sort(key=lambda x: [x["ts"], -(x["dur"]+x["ts"])])
t4 = time.time()
# Print sorting time for CPU events
debug_time_print(f"Sorted CPU events in {t4 - t3:.2f} seconds.")
# Build parent-child relationships for nested CPU events per thread
for (process_id, thread_id), events in cpu_events_by_thread.items():
    sorted_events = sorted(events, key=lambda x: [x["ts"], -(x["dur"]+x["ts"])])
    active_event_stack = []
    
    # Assign parent and children for each event
    for event in sorted_events:
        event["parent"] = None
        event["children"] = []
        
        while len(active_event_stack) > 0:
            potential_parent = active_event_stack[-1]
            event_end_time = event["ts"] + event["dur"]
            parent_end_time = potential_parent["ts"] + potential_parent["dur"]
            
            # Check if event fits within parent's duration
            if event["ts"] >= parent_end_time or event_end_time > parent_end_time:
                active_event_stack.pop()
            else:
                potential_parent["children"].append(event)
                event["parent"] = potential_parent
                break
        active_event_stack.append(event)

t5 = time.time()
# Print parent-child relationship building time
debug_time_print(f"Built parent-child relationships in {t5 - t4:.2f} seconds.")
def remove_module_id(name):
    idx = name.rfind("/")
    if idx != -1:
        name = name[idx+1:]
    if not "nn.Module: " in name: return name
    idx = name.rfind("_")
    if idx != -1:
        new_name = name[11:idx]
        #print(f"{name} --> {new_name}")

        return new_name
    return name

# Recursively build a dot-separated nested event name
def build_nested_event_name(event):
    if event.get("nested_key", None):
        return event["nested_key"]
    name = event["name"]
    name = remove_module_id(name)
    parent_event = event.get("parent", None)
    if parent_event:
        parent_name = build_nested_event_name(parent_event)
        nested_key = f"{parent_name}.{name}"
    else:
        nested_key = name
    event["nested_key"] = nested_key
    return nested_key

# Recursively compute nesting depth of an event
def get_event_depth(event):
    if "nesting_depth" in event:
        return event["nesting_depth"]
    parent_event = event.get("parent", None)
    if not parent_event:
        depth = 1
    else:
        depth = get_event_depth(parent_event) + 1
    event["nesting_depth"] = depth
    return depth

# Get total GPU execution time for an event (by external id)
def get_gpu_execution_time(event):
    if "args" in event and "External id" in event["args"]:
        external_id = event["args"]["External id"]
        if external_id in gpu_events_by_external_id:
            gpu_events = gpu_events_by_external_id[external_id]
            total_gpu_time = sum(gpu_event["dur"] for gpu_event in gpu_events)
            gpu_start_time = gpu_events[0]["ts"]
            gpu_end_time = gpu_events[-1]["ts"] + gpu_events[-1]["dur"]
            return total_gpu_time, gpu_start_time, gpu_end_time
    return 0, None, None

# Recursively calculate nested GPU time including children
def calculate_nested_gpu_time(event):
    gpu_time, gpu_start, gpu_end = get_gpu_execution_time(event)
    child_events = event.get("children", [])
    
    for child_event in child_events:
        child_gpu_time, child_start, child_end = calculate_nested_gpu_time(child_event)
        gpu_time += child_gpu_time
        if gpu_start is None or (child_start is not None and child_start < gpu_start):
            gpu_start = child_start
        if gpu_end is None or (child_end is not None and child_end > gpu_end):
            gpu_end = child_end
    
    return gpu_time, gpu_start, gpu_end

# Sum durations of all children of an event
def get_children_time(event):
    children_time = 0
    for child in event.get("children", []):
        children_time += child["dur"]
    return children_time

# Extract kernel info for GPU events associated with a CPU event
def extract_gpu_kernel_info(event):
    if "args" in event and "External id" in event["args"]:
        external_id = event["args"]["External id"]
        event_name = build_nested_event_name(event)
        if with_shape:
            input_dims = event.get("args", {}).get("Input Dims", "")
            input_dims = f"{input_dims}".replace(" ", "")
        else:
            input_dims = ""
        cpu_duration = event.get("dur", 0)
        if external_id in gpu_events_by_external_id:
            gpu_events = gpu_events_by_external_id[external_id]
            kernel_names = (gpu_event.get("name", "Unknown") for gpu_event in gpu_events)
            kernel_durations = (gpu_event.get("dur", 0) for gpu_event in gpu_events)
            return (event_name, input_dims), cpu_duration, kernel_names, kernel_durations
    return None, None, None, None

def get_operations(row):
    """
    Calculate the number of operations (FLOPs) for various GEMM-based operations.

    Supports:
    1. Matrix Multiplication (matmul): C = A × B
       - A: [M, K], B: [K, N], C: [M, N]
       - Operations = M × K × N

    2. Attention Mechanisms:
       - Q @ K^T: [B, H, S_q, D] @ [B, H, D, S_k] → [B, H, S_q, S_k]
         Operations = B × H × S_q × D × S_k
       - attn @ V: [B, H, S_q, S_k] @ [B, H, S_k, D] → [B, H, S_q, D]
         Operations = B × H × S_q × S_k × D
       - Total: B × H × S_q × D × (S_k + S_k) ≈ 2 × B × H × S_q × S_k × D
         For self-attention (S_q = S_k = S): 2 × B × H × S² × D

    Note: Each multiply-add (FMA) is counted as 1 operation here. The ×2 multiplier
    for FMA is applied later in the TFLOP/s calculation.

    Args:
        row: DataFrame row containing 'Shape' and 'EventName'

    Returns:
        int: Number of operations (FLOPs) for single execution, or None if not applicable
    """
    shape_str = row['Shape']
    event_name = row['EventName']

    try:
        shapes = ast.literal_eval(shape_str)

        # ========================================
        # 1. Standard Matrix Multiplication
        # ========================================
        if ("_C::weight_packed_linear" in event_name or "_C::onednn_mm" in event_name) and len(shapes) >= 3:
            # Extract unique dimension values from shape list
            unique_values = set()
            for shape in shapes:
                if isinstance(shape, list) and shape is not None:
                    unique_values.update(shape)
                else:
                    unique_values.add(shape)
            unique_values = sorted(list(unique_values))

            if len(unique_values) >= 3:
                # For matmul: [M, K, N] are the three dimensions
                # num_operations = M × K × N
                M, K, N = unique_values[0], unique_values[1], unique_values[2]
                return M * K * N

        # ========================================
        # 2. Attention Mechanisms
        # ========================================
        # Pattern: attention operations with Q, K, V tensors
        if any(keyword in event_name for keyword in [
            "attention", "attn", "sdpa", "scaled_dot_product"
        ]):
            # Try to extract attention dimensions
            # Common patterns:
            # - [[S,H,D], [S_k,H,D], [S_v,H,D], [S,H,D]] for Q,K,V,Output
            # - [[B,H,S,D], [B,H,S,D], [B,H,S,D], ...] for batched attention
            valid_shapes = []
            for shape in shapes:
                if isinstance(shape, list) and shape is not None and len(shape) >= 3:
                    valid_shapes.append(shape)

            if len(valid_shapes) >= 3:
                # Assume first 3 valid shapes are Q, K, V (or first 4 with output)
                q_shape = valid_shapes[0]
                k_shape = valid_shapes[1] if len(valid_shapes) > 1 else valid_shapes[0]
                v_shape = valid_shapes[2] if len(valid_shapes) > 2 else valid_shapes[0]

                if len(q_shape) == 3:
                    # vLLM-style attention: [S_q, H_q, D], [S_k, H_k, D], [S_v, H_v, D]
                    # Where:
                    # - S_q: query sequence length (new tokens)
                    # - S_k: key sequence length (may include KV cache)
                    # - H_q: number of query heads
                    # - H_k: number of key/value heads (may be < H_q for GQA)
                    S_q, H_q, D = q_shape[0], q_shape[1], q_shape[2]
                    S_k, H_k, _ = k_shape[0], k_shape[1], k_shape[2]

                    # For GQA (Grouped Query Attention):
                    # - If H_q > H_k: query heads are grouped, each group shares one K/V head
                    # - Each query head still processes full S_q × S_k × D operations
                    #
                    # Attention operations:
                    # 1. Q @ K^T: Each of H_q query heads attends to keys
                    #    Shape: [S_q, H_q, D] @ [S_k, H_k, D]^T
                    #    For GQA, query heads are replicated/grouped to match K/V
                    #    FLOPs = H_q × S_q × D × S_k
                    #
                    # 2. softmax: H_q × S_q × S_k (negligible, mostly memory-bound)
                    #
                    # 3. attn @ V: Attention weights times values
                    #    Shape: [H_q, S_q, S_k] @ [S_k, H_k, D]
                    #    FLOPs = H_q × S_q × S_k × D
                    #
                    # Total: H_q × S_q × D × S_k + H_q × S_q × S_k × D = 2 × H_q × S_q × S_k × D

                    num_operations = 2 * H_q * S_q * S_k * D
                    return num_operations

                elif len(q_shape) == 4:
                    # Standard batched attention: [B, H, S, D] or [B, S, H, D]
                    # Try to identify which dimension is which
                    dims = q_shape

                    # Heuristic: usually [B, H, S, D] where H is smallest, B is small, S and D vary
                    # Sort to identify: typically head_dim (64-256) < num_heads (8-32) < others
                    sorted_dims = sorted(enumerate(dims), key=lambda x: x[1])

                    # Common patterns:
                    # [B, H, S, D]: B small, H small, S large, D medium
                    # [B, S, H, D]: B small, S large, H small, D medium

                    if dims[1] < dims[0]:  # Likely [B, H, S, D]
                        B, H, S, D = dims[0], dims[1], dims[2], dims[3]
                    else:  # Likely [B, S, H, D]
                        B, S, H, D = dims[0], dims[1], dims[2], dims[3]

                    # For self-attention: 2 × B × H × S² × D
                    num_operations = 2 * B * H * (S ** 2) * D
                    return num_operations

        # ========================================
        # 3. Other GEMM-like operations (aten::mm, aten::addmm, etc.)
        # ========================================
        if "mm" in event_name.lower() or "matmul" in event_name.lower():
            # aten::mm: [[M, K], [K, N], [M, N]] - standard matrix multiplication
            # aten::addmm: [[M, N], [M, K], [K, N], [], [], [M, N]] - C = A @ B + bias
            #   where shapes are: [output, input, weight, ..., output]

            # Extract non-empty 2D shapes
            valid_shapes = []
            for shape in shapes:
                if isinstance(shape, list) and shape is not None and len(shape) == 2:
                    if shape[0] > 0 and shape[1] > 0:
                        valid_shapes.append(shape)

            if len(valid_shapes) >= 2:
                # For both mm and addmm, we need to identify M, K, N from the shapes
                # Strategy: Find the common dimension K (appears twice)
                # Then M and N are the other dimensions

                # Collect all dimensions with their frequency
                from collections import Counter
                dim_counts = Counter()
                for shape in valid_shapes:
                    dim_counts[shape[0]] += 1
                    dim_counts[shape[1]] += 1

                # Find K (the dimension that appears exactly twice - shared dimension)
                K = None
                for dim, count in dim_counts.items():
                    if count == 2:
                        K = dim
                        break

                if K is not None:
                    # Extract M and N (dimensions that appear once or are not K)
                    all_unique_dims = set()
                    for shape in valid_shapes:
                        all_unique_dims.add(shape[0])
                        all_unique_dims.add(shape[1])
                    all_unique_dims.discard(K)

                    if len(all_unique_dims) >= 2:
                        dims_list = sorted(list(all_unique_dims))
                        M, N = dims_list[0], dims_list[-1]  # smallest and largest remaining dims

                        # Calculate GEMM FLOPs
                        gemm_ops = M * K * N

                        # For addmm, add bias addition FLOPs (M * N element-wise additions)
                        if "addmm" in event_name.lower():
                            bias_ops = M * N
                            return gemm_ops + bias_ops
                        else:
                            return gemm_ops

                # Fallback: if we can't identify K uniquely, use heuristic
                # For typical GEMM: input [M, K], weight [K, N] -> output [M, N]
                if len(valid_shapes) >= 2:
                    # Use first two shapes to extract dimensions
                    shape1, shape2 = valid_shapes[0], valid_shapes[1]
                    # Try to find common dimension
                    if shape1[1] == shape2[0]:  # Standard case: [M, K] @ [K, N]
                        M, K, N = shape1[0], shape1[1], shape2[1]
                    elif shape1[0] == shape2[1]:  # Transposed case: [K, M] @ [N, K]
                        K, M, N = shape1[0], shape1[1], shape2[0]
                    elif shape1[0] == shape2[0]:  # Both start with same dim
                        M, K, N = shape1[0], shape1[1], shape2[1]
                    else:
                        # Last resort: extract unique values
                        unique_vals = sorted(list(set(shape1 + shape2)))
                        if len(unique_vals) >= 3:
                            M, K, N = unique_vals[0], unique_vals[1], unique_vals[2]
                        else:
                            return None

                    gemm_ops = M * K * N
                    if "addmm" in event_name.lower():
                        bias_ops = M * N
                        return gemm_ops + bias_ops
                    else:
                        return gemm_ops

        return None

    except Exception as e:
        return None


def get_num_weights(row):
    """
    Calculate the number of weight parameters for linear layers.

    For weight_packed_linear operations, the weight tensor is typically shapes[1].
    Weight shape is [K, N] where:
    - K: input dimension
    - N: output dimension

    Args:
        row: DataFrame row containing 'Shape' and 'EventName'

    Returns:
        int: Number of weight elements (K × N), or None if not applicable
    """
    shape_str = row['Shape']
    event_name = row['EventName']
    try:
        shapes = ast.literal_eval(shape_str)
        if ("_C::weight_packed_linear" in event_name) and (len(shapes) == 4 or len(shapes) == 3):
            # Weight tensor is typically at shapes[1]
            weight_shape = shapes[1]
            return weight_shape[0] * weight_shape[1]
        else:
            return None
    except Exception:
        return None


def get_total_tensor_movement(row):
    """
    Calculate total tensor movement in GB for various GEMM-based operations.

    Supports:
    1. Matrix Multiplication C = A × B:
       - Input A: [M, K] → M × K elements (read)
       - Weight B: [K, N] → K × N elements (read)
       - Output C: [M, N] → M × N elements (write)
       - Total: (M×K + K×N + M×N) × bytes_per_element

    2. Attention Mechanisms:
       - Q: [B, H, S, D] (read)
       - K: [B, H, S, D] or [B, H, S_cache, D] (read)
       - V: [B, H, S, D] or [B, H, S_cache, D] (read)
       - Attention weights: [B, H, S, S] (intermediate, read+write)
       - Output: [B, H, S, D] (write)
       - Total: 3×(B×H×S×D) + 2×(B×H×S²) for self-attention

    For BF16/FP16: bytes_per_element = 2 bytes

    This represents the minimum data that must move through memory hierarchy
    for a single execution of the operation.

    Args:
        row: DataFrame row containing 'Shape' and 'EventName'

    Returns:
        float: Total data movement in GB for single execution, or None if not applicable
    """
    shape_str = row['Shape']
    event_name = row['EventName']
    bytes_per_element = 2  # FP16/BF16 = 2 bytes per element

    try:
        shapes = ast.literal_eval(shape_str)

        # ========================================
        # 1. Standard Matrix Multiplication
        # ========================================
        if ("_C::weight_packed_linear" in event_name or "_C::onednn_mm" in event_name) and len(shapes) >= 3:
            # Extract unique dimensions from all shapes
            unique_values = set()
            for shape in shapes:
                if isinstance(shape, list) and shape is not None:
                    unique_values.update(shape)
                elif shape is not None:
                    unique_values.add(shape)

            if len(unique_values) >= 3:
                unique_values = sorted(list(unique_values))
                M, K, N = unique_values[0], unique_values[1], unique_values[2]

                # Calculate data movement for each tensor
                # Input tensor A: M × K elements
                input_size_bytes = M * K * bytes_per_element

                # Weight tensor B: K × N elements
                weight_size_bytes = K * N * bytes_per_element

                # Output tensor C: M × N elements
                output_size_bytes = M * N * bytes_per_element

                # Total movement in bytes, then convert to GB
                total_movement_bytes = input_size_bytes + weight_size_bytes + output_size_bytes
                total_movement_gb = total_movement_bytes / (1024**3)

                return total_movement_gb

        # ========================================
        # 2. Attention Mechanisms
        # ========================================
        if any(keyword in event_name for keyword in [
            "attention", "attn", "sdpa", "scaled_dot_product"
        ]):
            # Try to extract attention tensor shapes
            valid_shapes = []
            for shape in shapes:
                if isinstance(shape, list) and shape is not None and len(shape) >= 3:
                    valid_shapes.append(shape)

            if len(valid_shapes) >= 3:
                # Get Q, K, V shapes
                q_shape = valid_shapes[0]
                k_shape = valid_shapes[1] if len(valid_shapes) > 1 else valid_shapes[0]
                v_shape = valid_shapes[2] if len(valid_shapes) > 2 else valid_shapes[0]

                total_elements = 0

                if len(q_shape) == 3:
                    # vLLM-style attention: [S_q, H_q, D], [S_k, H_k, D], [S_v, H_v, D]
                    S_q, H_q, D = q_shape[0], q_shape[1], q_shape[2]
                    S_k, H_k, _ = k_shape[0], k_shape[1], k_shape[2]
                    S_v, H_v, _ = v_shape[0], v_shape[1], v_shape[2]

                    # Q read: S_q × H_q × D
                    q_read = S_q * H_q * D

                    # K read: S_k × H_k × D (may be smaller for GQA)
                    k_read = S_k * H_k * D

                    # V read: S_v × H_v × D (may be smaller for GQA)
                    v_read = S_v * H_v * D

                    # Attention scores: H_q × S_q × S_k (write + read)
                    # Each query head generates attention scores
                    attn_scores = 2 * H_q * S_q * S_k

                    # Output write: S_q × H_q × D
                    output_write = S_q * H_q * D

                    total_elements = q_read + k_read + v_read + attn_scores + output_write

                elif len(q_shape) == 4:
                    # Standard batched attention: [B, H, S, D]
                    dims = q_shape

                    # Determine layout (similar to operations function)
                    if dims[1] < dims[0]:  # Likely [B, H, S, D]
                        B, H, S, D = dims[0], dims[1], dims[2], dims[3]
                    else:  # Likely [B, S, H, D]
                        B, S, H, D = dims[0], dims[1], dims[2], dims[3]

                    # Q, K, V reads: 3 × B × H × S × D
                    qkv_reads = 3 * B * H * S * D

                    # Attention scores intermediate: B × H × S × S (write + read)
                    attn_scores = 2 * B * H * S * S

                    # Output write: B × H × S × D
                    output_write = B * H * S * D

                    total_elements = qkv_reads + attn_scores + output_write

                if total_elements > 0:
                    total_movement_bytes = total_elements * bytes_per_element
                    total_movement_gb = total_movement_bytes / (1024**3)
                    return total_movement_gb

        # ========================================
        # 3. Generic GEMM-like operations
        # ========================================
        if "mm" in event_name.lower() or "matmul" in event_name.lower():
            # Calculate total size of all tensors involved
            total_elements = 0
            for shape in shapes:
                if isinstance(shape, list) and shape is not None:
                    elements = 1
                    for dim in shape:
                        if dim > 0:
                            elements *= dim
                    total_elements += elements

            if total_elements > 0:
                # Approximate as: sum of all tensors
                total_movement_bytes = total_elements * bytes_per_element
                total_movement_gb = total_movement_bytes / (1024**3)
                return total_movement_gb

        return None

    except Exception as e:
        return None


# Print GPU kernel map if requested
if print_kernel_map:
    if has_gpu_events:
        print("GPU Kernel Map:")
        event_kernel_map = {}
        event_total_cpu_time = defaultdict(float)
        event_count = defaultdict(int)
        event_kernel_total_time = defaultdict(float)
        event_kernel_count = defaultdict(int)
        for event in all_cpu_events:
            event_key, cpu_time, kernel_names, kernel_durations = extract_gpu_kernel_info(event)
            if event_key is not None:
                event_total_cpu_time[event_key] += cpu_time
                event_count[event_key] += 1
                if event_key not in event_kernel_map:
                    event_kernel_map[event_key] = set()
                for kernel_name, kernel_duration in zip(kernel_names, kernel_durations):
                    event_kernel_map[event_key].add(kernel_name)
                    event_kernel_total_time[(event_key, kernel_name)] += kernel_duration
                    event_kernel_count[(event_key, kernel_name)] += 1

        # Print summary for each event and its kernels
        for event_key, kernels in event_kernel_map.items():
            if kernels:
                print(f" {event_total_cpu_time[event_key]/1000.0:9.3f} {event_count[event_key]:5d} {event_key[0]} {event_key[1]}")
                for kernel in kernels:
                    print(f" {event_kernel_total_time[(event_key, kernel)]/1000.0:9.3f} {event_kernel_count[(event_key, kernel)]:5d} --> {kernel}")
                print()
        exit(0)
    else:
        print("No GPU events found in the JSON data.")
        exit(0)

# Initialize summary dictionaries for durations and counts
sum_duration = {}
sum_self_duration = {}
sum_gpu_time = {}
count_events = {}

# Print event information
start_timestamp = all_cpu_events[0].get("ts", 0) if all_cpu_events else 0
previous_cpu_end = 0
previous_gpu_end = 0

# Print header if needed
if not summary_only and print_header:
    cpu_header = f"CPU: {'Start':>9} {'End':>9} {'Total':>6} {'Self':>6} {'Gap':>6}"
    if has_gpu_events:
        gpu_header = f"GPU: {'Start':>9} {'End':>9} {'Total':>6} {'Gap':>6}"
    else:
        gpu_header = ""
    
    print(f" {'Index':>5} {cpu_header}   {gpu_header}  {'EventName':<100} {'Shape'}")
    print("-" * 200)

# Iterate through all CPU events and print details or accumulate summary
for idx, event in enumerate(all_cpu_events):
    depth = get_event_depth(event)
    if max_depth > 0 and depth > max_depth:
        continue
    relative_timestamp = event.get("ts", "N/A") - start_timestamp
    duration = event.get("dur", "N/A")
    if depth == max_depth:
        self_duration = duration
    else:
        self_duration = duration - get_children_time(event)
    end_time = duration + relative_timestamp
    if event.get("parent", None) is None:
        gap = relative_timestamp - previous_cpu_end
        previous_cpu_end = end_time
    else:
        gap = 0
    nested_name = build_nested_event_name(event)
    nested_name = nested_name.replace(": ", ":").replace(" ", "_").replace("autograd::engine::evaluate_function:", "")
    input_dimensions = event.get("args", {}).get("Input Dims", "")
    input_dimensions = f"{input_dimensions}".replace(" ", "")
    external_id = event.get("args", {}).get("External id", "N/A")
    
    if has_gpu_events or kernel_only:
        # In kernel-only mode, use the kernel's own duration as GPU time
        if kernel_only and event.get("cat") == "kernel":
            gpu_execution_time = event.get("dur", 0)
            gpu_start_time = event.get("ts", start_timestamp)
            gpu_end_time = gpu_start_time + gpu_execution_time
        elif depth == max_depth or nested_gpu_time:
            gpu_execution_time, gpu_start_time, gpu_end_time = calculate_nested_gpu_time(event)
        else:
            gpu_execution_time, gpu_start_time, gpu_end_time = get_gpu_execution_time(event)

        if gpu_start_time is None:
            gpu_start_time = start_timestamp
        if gpu_end_time is None:
            gpu_end_time = start_timestamp
        
        relative_gpu_start = gpu_start_time - start_timestamp
        relative_gpu_end = gpu_end_time - start_timestamp

        if gpu_execution_time > 0 and (not nested_gpu_time or depth == 1):
            gpu_gap = relative_gpu_start - previous_gpu_end
            previous_gpu_end = relative_gpu_end
        else:
            gpu_gap = 0   
    
    if summary_only:
        # Accumulate summary statistics
        if with_shape:
            summury_key = (nested_name, input_dimensions)
        else:
            summury_key = (nested_name, "")

        if summury_key not in sum_duration:
            sum_duration[summury_key] = 0
        if summury_key not in sum_self_duration:
            sum_self_duration[summury_key] = 0
        if summury_key not in sum_gpu_time:
            sum_gpu_time[summury_key] = 0
        if summury_key not in count_events:
            count_events[summury_key] = 0
            
        count_events[summury_key] += 1
            
        sum_duration[summury_key] += duration
        sum_self_duration[summury_key] += self_duration
        if has_gpu_events or kernel_only:
            sum_gpu_time[summury_key] += gpu_execution_time
    
    else:
        # Print per-event details
        cpu_info = f"CPU: {relative_timestamp/1000.0:9.3f} {end_time/1000.0:9.3f} {duration/1000.0:6.3f} {self_duration/1000.0:6.3f} {gap/1000.0:6.3f}"
        if has_gpu_events:
            gpu_info = f"GPU: {relative_gpu_start/1000.0:9.3f} {relative_gpu_end/1000.0:9.3f} {gpu_execution_time/1000.0:6.3f} {gpu_gap/1000.0:6.3f}"
        else:
            gpu_info = ""
        print(f" {idx:5d} {cpu_info}   {gpu_info}  {nested_name:100s} {input_dimensions}")

# Print summary if requested
if summary_only:
    summary_events = []
    max_name_width = 15
    max_shape_width = 0
    for (nested_name, input_dims), total_duration in sum_duration.items():
        total_self_duration = sum_self_duration.get((nested_name, input_dims), 0)
        if has_gpu_events or kernel_only:
            total_gpu_time = sum_gpu_time.get((nested_name, input_dims), 0)
        else:
            total_gpu_time = 0
        count = count_events.get((nested_name, input_dims), 0)
        summary_events.append((nested_name, input_dims, count, total_self_duration, total_duration, total_gpu_time))
        max_name_width = max(max_name_width, len(nested_name))
        max_shape_width = max(max_shape_width, len(input_dims))
    max_name_width += 2
    # max_name_width = max_name_width if max_name_width <= 100 else 100
    # Sort summary events by GPU time, duration, and count
    summary_events.sort(key=lambda x: (x[-1], x[-2], x[-3]), reverse=True)
    if print_header:
        print(f"{'CPU_Total':>9} {'CPU_Self':>9} {'GPU_Total':>9}  {'Count':>5}  {'EventName':<{max_name_width}} {'Shape':<{max_shape_width}}")
        print("-" * (40+max_name_width+max_shape_width))
    for nested_name, input_dims, count, total_self_duration, total_duration, total_gpu_time in summary_events:
        truncated_name = nested_name if len(nested_name) <= max_name_width else "..."+nested_name[-max_name_width+3:]
        print(f"{total_duration/1000.0:9.3f} {total_self_duration/1000.0:9.3f} {total_gpu_time/1000.0:9.3f}  {count:5d}  {truncated_name:{max_name_width}s} {input_dims:<{max_shape_width}}")
    

    if args.output_excel_path:
        # ================================================================================
        # PERFORMANCE METRICS CALCULATION
        # ================================================================================
        #
        # This section calculates compute and memory performance metrics for profiled
        # operations, focusing on matrix multiplication (GEMM) kernels.
        #
        # KEY CONCEPTS:
        #
        # 1. TFLOP/s (Tera Floating-Point Operations per Second):
        #    - Measures computational throughput
        #    - For GEMM (C = A×B where A:[M,K], B:[K,N], C:[M,N]):
        #      * Base operations: M × K × N multiply-add pairs
        #      * FMA (Fused Multiply-Add) counts as 2 FLOPs
        #      * Total FLOPs = M × K × N × 2
        #    - Formula: TFLOP/s = (Total_FLOPs × Count) / (Time_ms × 1e9)
        #
        # 2. Bandwidth (GB/s):
        #    - Measures memory throughput
        #    - Accounts for all tensor movements: Input read + Weight read + Output write
        #    - For GEMM: Data_Movement = (M×K + K×N + M×N) × sizeof(dtype)
        #    - Formula: GB/s = (Total_Data_GB × Count × 1000) / Time_ms
        #
        # 3. Utilization %:
        #    - Compares achieved performance to theoretical peak
        #    - Helps identify compute-bound vs memory-bound operations
        #    - Utilization% = (Achieved / Peak) × 100
        #
        # TIME UNITS:
        #    - Input times are in microseconds (μs)
        #    - Converted to milliseconds (ms) for readability
        #    - Formulas use ms and convert to seconds where needed
        #
        # ================================================================================

        df = pd.DataFrame(summary_events, columns=['EventName', 'Shape', 'Count', 'CPU_Self', 'CPU_Total', 'GPU_Total'])
        # Convert microseconds to milliseconds for time columns
        df['CPU_Self'] = df['CPU_Self'] / 1000.0
        df['CPU_Total'] = df['CPU_Total'] / 1000.0
        df['GPU_Total'] = df['GPU_Total'] / 1000.0

        # ========================================
        # STEP 1: Calculate basic metrics
        # ========================================
        # Num_Weights: Number of weight parameters (K x N for matmul)
        df['Num_Weights'] = df.apply(get_num_weights, axis=1)

        # Num_Operations: Number of FLOPs for single operation (M x K x N for matmul)
        df['Num_Operations'] = df.apply(get_operations, axis=1)

        # Total_Tensor_Movement_GB: Data movement per single iteration in GB (input + weights + output)
        df['Total_Tensor_Movement_GB'] = df.apply(get_total_tensor_movement, axis=1)

        # ========================================
        # STEP 1b: Determine effective execution time (ms)
        # ========================================
        # Use GPU execution time when it is available (> 0), otherwise fall back to
        # CPU self time. GPU time reflects actual device kernel execution and is the
        # correct denominator for compute/bandwidth utilization on accelerators.
        # CPU self time (excluding children) is used when no GPU time was recorded.
        df['Effective_Time'] = df.apply(
            lambda row: row['GPU_Total'] if row['GPU_Total'] > 0 else row['CPU_Self'],
            axis=1
        )

        # ========================================
        # STEP 2: Calculate TFLOP/s (Compute Performance)
        # ========================================
        # Formula: TFLOP/s = (Num_Operations × 2 × Count) / (Effective_Time_ms × 1e9)
        #
        # Breakdown:
        # - Num_Operations: FLOPs for single matmul (M×K×N)
        # - ×2: Account for FMA (Fused Multiply-Add) in GEMM - each operation is a multiply + add
        # - ×Count: Total operations across all kernel invocations
        # - Effective_Time_ms: GPU time if available, else CPU self time (milliseconds)
        # - ÷1e9: Convert FLOP/ms to TFLOP/s
        #   * First ÷1000 converts ms to seconds: FLOP/ms → FLOP/s
        #   * Then ÷1e12 converts FLOP/s to TFLOP/s
        #   * Combined: ÷(1000 × 1e12 / 1000) = ÷1e9
        #
        # Result: Achieved TFLOP/s for this operation
        df['TFLOP/s'] = df.apply(
            lambda row: (row['Num_Operations'] * 2 * row['Count']) / (row['Effective_Time'] * 1e9)
            if row['Num_Operations'] is not None and row['Effective_Time'] > 0 else None,
            axis=1
        )

        # ========================================
        # STEP 3: Calculate Bandwidth (GB/s)
        # ========================================
        # Formula: Bandwidth (GB/s) = (Total_Tensor_Movement_GB × Count) / (Effective_Time_ms × 1e-3)
        #
        # Breakdown:
        # - Total_Tensor_Movement_GB: Data movement per iteration in GB (already accounts for ×2 bytes for bf16)
        # - ×Count: Total data movement across all kernel invocations
        # - Effective_Time_ms: GPU time if available, else CPU self time (milliseconds)
        # - ÷1e-3: Convert GB/ms to GB/s (divide by ms, multiply by 1000)
        #   * Actually: ÷(Effective_Time_ms / 1000) = ×(1000 / Effective_Time_ms)
        #
        # Simplified: Total_Movement_GB × Count × 1000 / Effective_Time_ms
        #
        # Result: Achieved memory bandwidth in GB/s
        df['Bandwidth (GB/s)'] = df.apply(
            lambda row: (row['Total_Tensor_Movement_GB'] * row['Count'] * 1000) / row['Effective_Time']
            if row['Total_Tensor_Movement_GB'] is not None and row['Effective_Time'] > 0 else None,
            axis=1
        )

        # ========================================
        # STEP 4: Calculate Total Metrics Across All Invocations
        # ========================================
        # Total data movement across all kernel calls (useful for memory footprint analysis)
        df['Total_Tensor_Movement_GB_All'] = df.apply(
            lambda row: row['Total_Tensor_Movement_GB'] * row['Count']
            if row['Total_Tensor_Movement_GB'] is not None else None,
            axis=1
        )

        # ========================================
        # STEP 5: Calculate Utilization Percentages
        # ========================================
        # TFLOPS Utilization: Percentage of theoretical peak compute achieved
        # Formula: (Achieved_TFLOPS / Peak_TFLOPS) × 100
        df['TFLOPS_Utilization_%'] = df.apply(
            lambda row: (row['TFLOP/s'] / args.peak_tflops * 100)
            if row['TFLOP/s'] is not None else None,
            axis=1
        )

        # Bandwidth Utilization: Percentage of theoretical peak memory bandwidth achieved
        # Formula: (Achieved_Bandwidth / Peak_Bandwidth) × 100
        df['Bandwidth_Utilization_%'] = df.apply(
            lambda row: (row['Bandwidth (GB/s)'] / args.peak_bandwidth * 100)
            if row['Bandwidth (GB/s)'] is not None else None,
            axis=1
        )

        # ========================================
        # STEP 6: Save to Excel
        # ========================================
        df.to_excel(args.output_excel_path, index=False)
        print(f"\n✓ Results exported to: {args.output_excel_path}")
        print(f"  - Total operations analyzed: {len(df)}")
        print(f"  - Operations with compute metrics: {df['TFLOP/s'].notna().sum()}")
        print(f"  - Operations with bandwidth metrics: {df['Bandwidth (GB/s)'].notna().sum()}")

        # Print summary statistics
        if df['TFLOP/s'].notna().sum() > 0:
            avg_tflops = df['TFLOP/s'].mean()
            max_tflops = df['TFLOP/s'].max()
            print(f"\n  Performance Summary:")
            print(f"  - Average TFLOP/s: {avg_tflops:.2f}")
            print(f"  - Max TFLOP/s: {max_tflops:.2f}")

            # Note about high TFLOP/s values
            if max_tflops > args.peak_tflops * 2:
                print(f"\n  ⚠ Note: Very high TFLOP/s values (>{args.peak_tflops*2:.0f}) detected.")
                print(f"     This typically occurs with attention operations using large KV caches,")
                print(f"     where S_k (key sequence length) includes the entire cache.")
                print(f"     The computation is: 2 × H × S_q × S_k × D FLOPs")
                print(f"     Large S_k with efficient caching can achieve high throughput.")  

     

# print("Processing complete.")