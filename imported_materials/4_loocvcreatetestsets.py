import pandas as pd
import os
import sys
import numpy as np
import glob

def create_loocv_testsets(input_file, output_dir):
    """
    Create leave-one-out cross-validation test sets from sample data
    """
    print(f"Processing: {input_file}")
    
    # Read the sample data
    data = pd.read_csv(input_file, sep='\t')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating LOOCV test sets for {len(data)} samples in: {output_dir}")
    
    # Create a test set for each sample
    for index, row in data.iterrows():
        # Create a copy of the data
        modified_data = data.copy()
        
        # Mask the location and environmental data for this sample
        sample_id = row['sampleID']
        
        # Handle cases where some columns might not exist
        cols_to_mask = []
        for col in ['y', 'x', 'cov1', 'cov2', 'cov3']:
            if col in modified_data.columns:
                cols_to_mask.append(col)
        
        modified_data.loc[index, cols_to_mask] = [np.nan] * len(cols_to_mask)
        
        # Create output filename with naming convention: masked_data_sample_<>.tsv
        output_file = os.path.join(output_dir, f"masked_data_{sample_id}.tsv")
        
        # Save the modified data
        modified_data.to_csv(output_file, sep='\t', index=False)
        
        if index % 100 == 0:
            print(f"  Processed {index + 1} samples")

    print(f"  Created {len(data)} test files in {output_dir}")

def process_all_full_files(input_dir, base_output_dir):
    """
    Process all full sample_data files (not samp1000 versions) in the input directory
    """
    # Find all sample_data files that don't contain 'samp1000'
    all_sample_files = glob.glob(os.path.join(input_dir, "sample_data_*.tsv"))
    full_files = [f for f in all_sample_files if 'samp1000' not in f and 'samp500' not in f and 'samp100' not in f]
    
    if not full_files:
        print(f"No full sample_data files found in {input_dir}")
        print(f"Looking for files like: sample_data_rn_laSLiM_map9_h0_seed50_d0.5.tsv")
        return
    
    print(f"Found {len(full_files)} full sample_data files to process:")
    for file in full_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    for samp_file in full_files:
        # Extract base name for creating testset directory
        basename = os.path.basename(samp_file)
        # Remove 'sample_data_' prefix and '.tsv' suffix to get the genotype identifier
        # sample_data_rn_laSLiM_map9_h0_seed50_d0.5.tsv -> rn_laSLiM_map9_h0_seed50_d0.5
        dataset_name = basename.replace('sample_data_', '').replace('.tsv', '')
        
        # Create testset directory for this dataset (reflects the genotype matrix)
        testset_dir = os.path.join(base_output_dir, f"testsets_{dataset_name}")
        
        # Create leave-one-out test sets
        create_loocv_testsets(samp_file, testset_dir)
    
    print(f"\nAll test sets created in: {base_output_dir}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        
        # Check if input is a directory
        if os.path.isdir(input_dir):
            # Process all full sample_data files in directory
            process_all_full_files(input_dir, output_dir)
        else:
            print(f"Error: {input_dir} is not a valid directory")
            sys.exit(1)
    else:
        print("Usage: python 4_loocvcreatetestsets.py <input_directory> <output_directory>")
        print("Example:")
        print("  python 4_loocvcreatetestsets.py DEBUG/ DEBUG/testsets")
        print("")
        print("This will process all full sample_data files (not samp1000 versions) and create LOOCV test sets")
        print("with directory names reflecting their corresponding genotype matrices.")
        sys.exit(1)
