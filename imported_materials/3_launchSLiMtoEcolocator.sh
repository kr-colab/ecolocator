#!/bin/bash

# SLiM to Ecolocator launcher script
# Usage: bash 3_launchSLiMtoEcolocator.sh
# Processes all rn_*.trees files in the OUTDIR directory

OUTDIR="slimout_8x8s5d1g123"

if [ ! -d "$OUTDIR" ]; then
    echo "Error: Directory $OUTDIR not found"
    exit 1
fi

echo "Processing rn_*.trees files in $OUTDIR for Ecolocator conversion"

# Find all rn_*.trees files in the OUTDIR and submit jobs
for TREE_FILE in "$OUTDIR"/rn_*.trees; do
    if [ -f "$TREE_FILE" ]; then
        echo "Submitting SLiM to Ecolocator job for: $TREE_FILE"
        sbatch ./runSLiMtoEcolocator.sh "$TREE_FILE"
    fi
done


