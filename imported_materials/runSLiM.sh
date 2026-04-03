#!/bin/bash
#SBATCH --account=kernlab
#SBATCH --partition=kern
#SBATCH --job-name=SLiM
#SBATCH --time=8:00:00
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --error=job%j.err
#SBATCH --out=job%j.out

GENARC=$1
MAP_FILE=$2
MAP_ID=$3
OUTDIR=./slimout_8x8s5d1g123/

if [ -z "$GENARC" ] || [ -z "$MAP_FILE" ] || [ -z "$MAP_ID" ]; then
    echo "Usage: sbatch ./runSLiM.sh <genarc> <map_file> <map_id>"
    exit 1
fi

echo "Running SLiM with GENARC=$GENARC, MAP=$MAP_FILE"

# Create output directory if it doesn't exist
mkdir -p $OUTDIR

# Run SLiM with parameters
slim -d "GENARC=$GENARC" \
     -d "ENV_MAP='$MAP_FILE'" \
     -d "OUTFILE='$OUTDIR/laSLiM_${MAP_ID}_g${GENARC}.trees'" \
     localadaptation_nonWF.slim