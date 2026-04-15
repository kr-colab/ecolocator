import allel
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vcf", help="path to VCF (or .vcf.gz)")
parser.add_argument("--out", help="path for matrix TSV output")
args = parser.parse_args()

vcf = allel.read_vcf(args.vcf)
genotypes = allel.GenotypeArray(vcf["calldata/GT"])
samples = vcf["samples"]

# sum haplotypes per sample to get 0/1/2 dosage counts
dosage = genotypes.to_n_alt()  # shape: (n_variants, n_samples)

# build dataframe: rows=samples, columns=SNP positions
positions = vcf["variants/POS"]
col_names = [str(p) for p in positions]
df = pd.DataFrame(dosage.T, columns=col_names)
df.insert(0, "sampleID", samples)

df.to_csv(args.out, sep="\t", index=False)
print(f"Wrote matrix to {args.out}")
