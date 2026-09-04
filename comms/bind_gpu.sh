#!/usr/bin/env bash
#
# Rank-local GPU pinning helper for the CCL benchmark tool.
# Intel MPI exports MPI_LOCALRANKID (0-based local rank on the node); mapping
# it 1:1 to ZE_AFFINITY_MASK gives each rank exclusive use of one GPU/tile.
#
if [[ -n "${MPI_LOCALRANKID:-}" ]]; then
    export ZE_AFFINITY_MASK="${MPI_LOCALRANKID}"
fi
exec "$@"
