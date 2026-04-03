#!/bin/bash

# for DISPERSAL in 0.1 0.3 0.5; do
#     for REP in {1..2}; do
#         sbatch ./runSLiM.sh $DISPERSAL $REP
#     done
# done

for ; do
    for MAP_FILE in ./extraMAPS8x8/*.png; do
        # Extract just the filename without path
        MAP_NAME=$(basename "$MAP_FILE")
        # Remove the .png extension for cleaner naming
        MAP_ID=$(basename "$MAP_FILE" .png)
        
        # Single run per map (no replicates)
        sbatch ./runSLiM.sh $GENARC "$MAP_FILE" "$MAP_ID"
    done
done