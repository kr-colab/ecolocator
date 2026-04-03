import tskit
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def process_slim_output(tree_file, output_dir="./slimout_8x8s5d1g123/"):
    """
    Convert SLiM tree sequence output to Ecolocator-compatible format
    """
    print(f"Processing {tree_file}")
    
    # Load tree sequence
    ts = tskit.load(tree_file)
    sample_nodes = ts.samples()

    # Extract base name from input file (remove path and extension)
    base_name = os.path.splitext(os.path.basename(tree_file))[0]
    
    # Extract genotype matrix
    print("Extracting genotype matrix...")
    metadata = ts.metadata['SLiM']['user_metadata']
    genotype_matrix = ts.genotype_matrix().T

    # Map nodes to individuals
    node_to_individual = np.zeros(ts.num_nodes, dtype=int) - 1
    for individual in ts.individuals():
        for node in individual.nodes:
            node_to_individual[node] = individual.id

    # Create individual genotype matrix
    num_individuals = ts.num_individuals
    num_sites = ts.num_sites
    individual_genotype_matrix = np.zeros((num_individuals, num_sites), dtype=int)

    for node_index, individual_id in enumerate(node_to_individual):
        if individual_id != -1:  # ignore nodes not associated with any individual
            individual_genotype_matrix[individual_id] += genotype_matrix[node_index]

    # Clip to 0-2 (diploid)
    individual_genotype_matrix = np.clip(individual_genotype_matrix, 0, 2)

    # Create labels
    individual_labels = [f"sample_{i}" for i in range(num_individuals)]
    variant_labels = [f"variant_{i}" for i in range(num_sites)]

    # Create genotype dataframe
    genotype_df = pd.DataFrame(individual_genotype_matrix, 
                             index=individual_labels, 
                             columns=variant_labels)
    genotype_df.index.name = 'sampleID'

    # Save full genotype matrix
    geno_file = os.path.join(output_dir, f"geno_{base_name}.tsv")
    genotype_df.to_csv(geno_file, sep='\t')
    print(f"Saved genotype matrix: {geno_file}")

    # Extract sample data
    print("Extracting sample data...")
    individual_data = []
    for individual in ts.individuals():
        if individual.id in sample_nodes:
            location = individual.location
            optimum = metadata['optima'][individual.id]
            if len(location) == 3: 
                lat, lon, x = location 
                individual_data.append((f'sample_{individual.id}', lat, lon, optimum))

    # Create sample dataframe
    df_individuals = pd.DataFrame(individual_data, 
                                columns=['sampleID', 'y', 'x', 'cov1'])

    # Save full sample data
    sample_file = os.path.join(output_dir, f"sample_data_{base_name}.tsv")
    df_individuals.to_csv(sample_file, sep='\t', index=False)
    print(f"Saved sample data: {sample_file}")

    # Create sampled version (1000 samples)
    print("Creating sampled version...")
    common_samples = genotype_df.index.intersection(df_individuals['sampleID'])
    
    if len(common_samples) >= 1000:
        np.random.seed(42)  
        sampled_ids = np.random.choice(common_samples, size=1000, replace=False)
        
        sampled_genotype_df = genotype_df.loc[sampled_ids]
        sampled_individuals_df = df_individuals[df_individuals['sampleID'].isin(sampled_ids)]
        
        sampled_genotype_df = sampled_genotype_df.sort_index()
        sampled_individuals_df = sampled_individuals_df.sort_values(by='sampleID')
        
        # Save sampled versions with samp1000 suffix
        sampled_geno_file = os.path.join(output_dir, f"geno_{base_name}samp1000.tsv")
        sampled_sample_file = os.path.join(output_dir, f"sample_data_{base_name}samp1000.tsv")
        
        sampled_genotype_df.to_csv(sampled_geno_file, sep='\t')
        sampled_individuals_df.to_csv(sampled_sample_file, sep='\t', index=False)
        
        print(f"Saved sampled genotype matrix: {sampled_geno_file}")
        print(f"Saved sampled sample data: {sampled_sample_file}")
    else:
        print(f"Warning: Only {len(common_samples)} samples available, skipping 1000-sample subset")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python SLiMtoEcolocator.py <tree_file> [output_dir]")
        sys.exit(1)
    
    tree_file = sys.argv[1]
    output_dir = sys.argv[2] 
    
    process_slim_output(tree_file, output_dir)
