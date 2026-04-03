#!/bin/bash
#SBATCH --account=kernlab
#SBATCH --partition=kern
#SBATCH --job-name=SLiMtoEco
#SBATCH --time=6:00:00
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --error=job%j.err
#SBATCH --out=job%j.out

# SLiM to Ecolocator converter runner script
# Usage: sbatch ./runSLiMtoEcolocator.sh <input_trees_file>

INPUT_FILE=$1

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: sbatch ./runSLiMtoEcolocator.sh <input_trees_file>"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Tree file $INPUT_FILE not found"
    exit 1
fi

# Extract directory for output (same as input directory)
INPUT_DIR=$(dirname "$INPUT_FILE")

echo "Converting SLiM output to Ecolocator format: $INPUT_FILE"
echo "Output directory: $INPUT_DIR"

# Run the Python script
python SLiMtoEcolocator.py "$INPUT_FILE" "$INPUT_DIR"

echo "SLiM to Ecolocator conversion complete for $INPUT_FILE"
