# Command-line usage

`EcoLocator` needs two inputs:

- **Genotypes** -- a `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV
  (`sampleID` plus one column per SNP, entries `0`/`1`/`2`).
- **Sample metadata** -- a TSV with columns `sampleID`, `x`, `y`, and any number
  of environmental covariate columns. Use `NA` for the `x`, `y`, and covariate
  values of samples whose location and environment should be predicted.

Please find example data on Github at 
`./data/test_genotypes.vcf.gz` and `./data/test_sample_data_cov.txt`



Subcommands:

- `ecolocator train` — fit a model and save it to an output directory.
- `ecolocator predict` — predict locations and covariates for samples with unknown coordinates, using a saved model.
- `ecolocator loo` — leave-one-out cross-validation: iteratively hold out each sample, train, and predict.
- `ecolocator attribute` — attribute SNP importance for predicted samples using SHAP.

---

## `ecolocator train`

Trains an `EcoLocator` model on a genotype file plus a sample-metadata TSV and
writes the fitted model (`model.keras`, `arrays.npz`, `params.json`) to a
directory:

```bash
ecolocator train \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/my_model/
```

Genotype input may be `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV. The
sample-metadata TSV must have columns `sampleID`, `x`, `y`, plus one column per
environmental covariate. Leave `x`/`y` blank (or `NA`) for samples whose
location is to be predicted.

Hyperparameter and training flags: `--nlayers`, `--width`, `--dropout-prop`,
`--loc-weight`, `--env-weight`, `--cov-transforms` (comma-separated
per-covariate, e.g. `none,log,log`), `--max-epochs`, `--patience`,
`--batch-size`, `--min-mac`, `--max-snps`, `--train-split`, `--seed`,
`--verbose`. Run `ecolocator train --help` for the full list.

---

## `ecolocator predict`

Loads a saved model and predicts locations and covariates for samples whose
`x`/`y` are missing in the sample-metadata file. Pass the **same masked
sample-data file** to both `train` and `predict` — `train` ignores the masked
rows during fitting, and `predict` returns predictions only for those rows:

```bash
ecolocator predict \
    --model out/my_model/ \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/predictions.tsv
```

Output is a tab-separated file with columns `sampleID`, `x`, `y`, and one
column per covariate.

---

## `ecolocator loo`

Runs leave-one-out cross-validation over all samples. Pass the **full**
sample-metadata file (all coordinates known) — the method handles masking
internally, holding out one sample per fold, training a model, and predicting
that sample. Results are checkpointed after every fold:

```bash
ecolocator loo \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/loo_predictions.tsv
```

Use `--max-folds N` to run only the first N folds, useful for a quick
smoke-test before committing to a full run. All training hyperparameter flags
are identical to `train`.

---

## `ecolocator attribute`

Loads a saved model and computes SNP importance scores for samples with unknown
coordinates using SHAP. Pass the **same masked sample-data file** as used for
`predict` — rows with known coordinates form the background for SHAP, rows with
missing `x`/`y` are the samples being attributed:

```bash
ecolocator attribute \
    --model out/my_model/ \
    --genotypes data/genotypes.tsv \
    --sample-data data/test_sample_data_cov.txt \
    --out out/attributions.tsv
```

Output is a tab-separated file with one row per SNP and one column per output
variable (`x`, `y`, and each covariate), containing mean absolute SHAP values
across all attributed samples. The `snp_id` column identifies each SNP using
whatever real identifier was available in `--genotypes`: the VCF `ID` field
(falling back to `CHROM:POS` for every SNP if any are missing an ID, so
identifiers don't mix formats), the zarr store's equivalent fields, or the
genotype matrix's own column headers.

Use `--background-size` (default 100) to control how many training samples are
used as the SHAP background — larger values are more stable but slower. Use
`--min-maf` to exclude low-frequency SNPs from the output. Add `--save-raw` to
also write per-sample SHAP values to `<out-stem>_raw.tsv` alongside the summary.

