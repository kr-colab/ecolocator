#!/bin/bash
#SBATCH --account=kernlab
#SBATCH --partition=kern
#SBATCH --job-name=msprime
#SBATCH --time=4:00:00
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --error=job%j.err
#SBATCH --out=job%j.out

# msprime neutral overlay runner script
# Usage: sbatch ./runmsprime.sh <input_trees_file>

INPUT_FILE=$1

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: sbatch ./runmsprime.sh <input_trees_file>"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Tree file $INPUT_FILE not found"
    exit 1
fi

# Extract directory and filename for output
INPUT_DIR=$(dirname "$INPUT_FILE")
BASENAME=$(basename "$INPUT_FILE" .trees)
OUTPUT_FILE="${INPUT_DIR}/rn_${BASENAME}.trees"

echo "Running msprime neutral overlay on: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# Run the Python script with input and output parameters
python LASLiM_neutralmoverlay.py "$INPUT_FILE" "$OUTPUT_FILE"

echo "msprime processing complete for $INPUT_FILE"
