


## Runs
```
source /data/swtools/intel/oneapi/2026.0/oneapi-vars.sh
cd general-benchmarks/comms
./run_benchmark.sh --backend cpu --devices 4 --ops allreduce,allgather --sizes 4096,1048576,16777216
./run_benchmark.sh --backend xpu --devices 2 --ops allreduce --sizes 1048576,16777216   # run on a GPU node

# All 4 GPUs on a node, sweeping sizes to find achieved PCIe allreduce bandwidth:
./run_benchmark.sh --backend xpu --devices 4 --ops allreduce \
    --sizes 4096,65536,1048576,16777216,134217728 --iters 20 --warmup-iters 5
```

Read the `busbw[GB/s]` column for the achieved bus bandwidth (bandwidth-corrected for
the algorithm, i.e. comparable across ring/recursive-doubling implementations). GPUs
that only share PCIe (no Xe-Link) will show a oneCCL warning
`topology recognition shows PCIe connection between devices` confirming the fabric used.