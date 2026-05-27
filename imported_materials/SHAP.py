#!/usr/bin/env python3
"""
SHAP Analysis Script for Ecolocator Results
Converts notebook cells 1, 2, 3, 5, 6, 7 from SHAP2.ipynb into a command-line script.
Only generates 5% MAF cutoff files.
"""

import glob
import re
import os
import argparse
import sys

# Import required libraries (Cell 1)
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def euclid_loss(y_true, y_pred):
    """Custom loss function for tensorflow models (Cell 5)"""
    import tensorflow.keras.backend as K
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def filter_common_snps(snp_ids, train_data, min_frequency=0.05):
    """
    Filter SNPs to keep only those with allele frequency >= min_frequency (Cell 6)
    
    Args:
        snp_ids: array of SNP IDs
        train_data: training data array (samples x SNPs)
        min_frequency: minimum allele frequency threshold (default 0.05 = 5%)
    
    Returns:
        filtered_snp_ids, filtered_indices, filtered_train_data
    """
    # Calculate allele frequencies (assuming 0,1,2 encoding)
    allele_freqs = np.mean(train_data, axis=0) / 2.0
    
    # Find SNPs that meet frequency criteria
    common_mask = (allele_freqs >= min_frequency) & (allele_freqs <= (1 - min_frequency))
    
    print(f"Filtering SNPs:")
    print(f"  Total SNPs: {len(snp_ids)}")
    print(f"  Common SNPs (freq >= {min_frequency}): {np.sum(common_mask)}")
    print(f"  Filtered out: {len(snp_ids) - np.sum(common_mask)}")
    
    # Filter data
    filtered_snp_ids = snp_ids[common_mask]
    filtered_indices = np.where(common_mask)[0]
    filtered_train_data = train_data[:, common_mask]
    
    return filtered_snp_ids, filtered_indices, filtered_train_data


def analyze_shap_values(base_dir, output_dir, neutrality_file=None):
    """
    Main SHAP analysis function (adapted from Cell 7)
    
    Args:
        base_dir: Directory containing the model and data files
        output_dir: Directory to save SHAP results
        neutrality_file: Optional neutrality file (not used in current implementation)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load neutrality information if provided (Cell 3 logic - adapted but not required)
    neutrality_lookup = {}
    if neutrality_file and os.path.exists(neutrality_file):
        print(f"Loading neutrality information from: {neutrality_file}")
        neutrality_df = pd.read_csv(neutrality_file)
        neutrality_lookup = dict(zip(neutrality_df['variant_id'], neutrality_df['classification']))
        print(f"Loaded neutrality info for {len(neutrality_lookup)} variants")
    
    # Find all model files and process them (Cell 7 main loop)
    model_pattern = os.path.join(base_dir, "sample_*_model.keras")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"Error: No model files found matching pattern: {model_pattern}")
        return False
    
    print(f"Found {len(model_files)} model files to process")
    
    processed_count = 0
    for model_path in model_files:
        # Extract the sample ID (whatever sits between "sample_" and "_model.keras")
        m = re.search(r"sample_(.*?)_model\.keras$", model_path)
        if not m:
            print(f"Warning: Could not parse sample ID from {model_path}")
            continue
        sample_id = m.group(1)
        
        # Build the other per-sample file paths
        snp_path = os.path.join(base_dir, f"sample_{sample_id}_kept_snp_ids.txt")
        train_path = os.path.join(base_dir, f"sample_{sample_id}_traingen_array.npy")
        pred_path = os.path.join(base_dir, f"sample_{sample_id}_predgen_array.npy")
        pred_samps = os.path.join(base_dir, f"sample_{sample_id}_kept_pred_samples.txt")
        
        # Check if all required files exist
        required_files = [snp_path, train_path, pred_path, pred_samps]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"Warning: Missing files for sample {sample_id}: {missing_files}")
            continue
        
        try:
            # 1) Load everything for this sample
            print(f"Processing sample {sample_id}...")
            snp_ids = np.loadtxt(snp_path, dtype=str)
            X_train = np.load(train_path)
            X_pred = np.load(pred_path)
            
            # 2) Filter for common alleles (>5% frequency) - for analysis focus only
            filtered_snp_ids, common_indices, filtered_X_train = filter_common_snps(
                snp_ids, X_train, min_frequency=0.05
            )
            
            # 3) Continue with ORIGINAL unfiltered data for SHAP
            pred_sample = np.loadtxt(pred_samps, dtype=str, ndmin=1)[0]
            i = 0
            x_i = X_pred[i].reshape(1, -1)  # Use original X_pred (all SNPs)
            
            # 4) Build background from original train set
            bg_size = 100
            idx = np.random.choice(X_train.shape[0], bg_size, replace=False)
            bg = X_train[idx, :]  # Use original X_train (all SNPs)
            
            # 5) Load & wrap the model heads (standard approach)
            model = tf.keras.models.load_model(model_path, 
                           custom_objects={"Custom>euclid_loss": euclid_loss})
            model_loc = tf.keras.Model(inputs=model.input, outputs=model.output[0])
            model_env = tf.keras.Model(inputs=model.input, outputs=model.output[1])
            
            # 6) Compute SHAP on all SNPs (as the model expects)
            print(f"  Computing SHAP values...")
            expl_loc = shap.GradientExplainer(model_loc, bg)
            expl_env = shap.GradientExplainer(model_env, bg)
            shap_loc = expl_loc.shap_values(x_i)
            shap_env_v = expl_env.shap_values(x_i)
            
            # 7) Create series for ALL SNPs first
            shap_lat_full = pd.Series(shap_loc[0, :, 0], index=snp_ids, name=f"{sample_id}_lat")
            shap_lon_full = pd.Series(shap_loc[0, :, 1], index=snp_ids, name=f"{sample_id}_lon")
            shap_env_full = pd.Series(shap_env_v[0, :, 0], index=snp_ids, name=f"{sample_id}_env")
            
            # 8) Filter the SHAP results to focus on common SNPs
            shap_lat = shap_lat_full.iloc[common_indices]
            shap_lon = shap_lon_full.iloc[common_indices]
            shap_env = shap_env_full.iloc[common_indices]
            
            # 9) Save filtered results (5% MAF cutoff only)
            print(f"  Saving results...")
            shap_lat.to_csv(os.path.join(output_dir, f"shap_lat_filtered5pct_{sample_id}.csv"))
            shap_lon.to_csv(os.path.join(output_dir, f"shap_lon_filtered5pct_{sample_id}.csv"))
            shap_env.to_csv(os.path.join(output_dir, f"shap_env_filtered5pct_{sample_id}.csv"))
            
            processed_count += 1
            print(f"  ✓ Done sample {sample_id}: {len(snp_ids)} total SNPs, {len(filtered_snp_ids)} common SNPs analyzed")
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            continue
    
    print(f"Successfully processed {processed_count} samples")
    return processed_count > 0


def main():
    parser = argparse.ArgumentParser(description='Analyze SHAP values for Ecolocator results')
    parser.add_argument('base_dir', help='Base directory containing model and data files')
    parser.add_argument('output_dir', help='Output directory for SHAP results')
    parser.add_argument('--neutrality-file', help='Optional neutrality CSV file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}")
        sys.exit(1)
    
    print("=" * 50)
    print("SHAP Analysis for Ecolocator Results")
    print("=" * 50)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.neutrality_file:
        print(f"Neutrality file: {args.neutrality_file}")
    print("=" * 50)
    
    # Run the analysis
    success = analyze_shap_values(args.base_dir, args.output_dir, args.neutrality_file)
    
    if success:
        print("Analysis completed successfully!")
        sys.exit(0)
    else:
        print("Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()