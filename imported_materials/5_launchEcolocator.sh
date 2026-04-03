GENO_DIR="slimout_8x8s5d1g123"
TESTSET_DIR="slimout_8x8s5d1g123/testsets"
OUTPUT_DIR="out_8x8s5d1g123"

# Create main output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

#  count=0

# Loop through all full genotype files (not downsampled)
for geno_file in "$GENO_DIR"/geno_rn_laSLiM_*.tsv; do
    # Skip downsampled files (those with 'samp' in filename)
    if [[ "$geno_file" == *"samp"* ]]; then
        continue
    fi
    
    # Extract the unique identifier from the filename
    base_name=$(basename "$geno_file")
    # For files like: geno_rn_laSLiM_map1_d0.1.tsv
    if [[ $base_name =~ geno_rn_laSLiM_(.*)\.tsv ]]; then
        identifier="${BASH_REMATCH[1]}"  # e.g., map1_d0.1
    else
        echo "Could not parse filename: $base_name"
        continue
    fi
    
    # Construct the matching testset directory
    testset_subdir="$TESTSET_DIR/testsets_rn_laSLiM_${identifier}"

    # Define the trained model file path (if using trained models)
    #trained_model="$GENO_DIR/outmodels_noperm/noperm_${identifier}" 
    
    # Ensure the testset directory exists
    if [ -d "$testset_subdir" ]; then
        # Loop through all masked* files in the testset directory
        for test_file in "$testset_subdir"/masked_data_*.tsv; do
            test_base=$(basename "$test_file")
            ###sample_index=$(echo "$test_base" | sed -E 's/masked_data_sample_([0-9]+)\.tsv$/\1/')
            ###model_file="$trained_model/sample_${sample_index}_model.keras"
            
            ### Define the output directory for this run
            out_subdir="$OUTPUT_DIR/results_${identifier}/"
            mkdir -p "$out_subdir"  

            echo "$geno_file $test_file $out_subdir"
            ##echo "$geno_file $test_file $out_subdir $model_file"

            ###    count=$((count + 1))
            ###  if [ "$count" -ge 5 ]; then
            ###      break 2  # exit both loops
            ###  fi
        done
    fi
done >joblist.txt

LINES_PER_CHUNK=10
split -l $LINES_PER_CHUNK joblist.txt joblist_temp_
# will create job files for each worker, named joblist_temp_aa, joblist_temp_ab, etc.
for jobfile in joblist_temp_*; do
    sbatch runEcolocator.sh $jobfile # for gametime
    #bash runEcolocator.sh $jobfile  # for testing
done
#rm -f joblist_temp_*  # remove any leftover files
