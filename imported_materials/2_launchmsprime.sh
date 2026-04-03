#!/bin/bash

# msprime neutral overlay launcher script
# Usage: bash 2_launchmsprime.sh
# Processes all .trees files in the OUTDIR

OUTDIR="slimout_8x8s5d1g123"

if [ ! -d "$OUTDIR" ]; then
    echo "Error: Directory $OUTDIR not found"
    exit 1
fi

echo "Processing .trees files in ${OUTDIR}"

# Find all .trees files in the OUTDIR and submit jobs
for TREE_FILE in "$OUTDIR"/*.trees; do
    if [ -f "$TREE_FILE" ]; then
        echo "Submitting msprime job for: $TREE_FILE"
        sbatch ./runmsprime.sh "$TREE_FILE"
    fi
done
