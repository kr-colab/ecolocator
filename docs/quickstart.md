# Quickstart

This page shows the Python API. For the equivalent command-line workflow, see
{doc}`cli`.

## Inputs

`ecoLocator` needs two inputs:

- **Genotypes** -- a `.vcf`, `.vcf.gz`, `.zarr`, or a dosage-matrix TSV
  (`sampleID` plus one column per SNP, entries `0`/`1`/`2`).
- **Sample metadata** -- a TSV with columns `sampleID`, `x`, `y`, and any number
  of environmental covariate columns. Use `NA` for the `x`/`y` of samples whose
  location should be predicted.

## Train a model

```python
from ecolocator import EcoLocator

model = EcoLocator()
model.fit(
    genotype_path="genotypes.vcf.gz",
    sample_data_path="samples.tsv",
)
model.save("my_model")
```

`save()` writes a directory containing `model.keras`, `arrays.npz`, and
`params.json`.

## Predict

Reload a saved model and predict coordinates and covariates for the samples whose
`x`/`y` are `NA`:

```python
from ecolocator import EcoLocator

model = EcoLocator.load("my_model")
predictions = model.predict(
    genotype_path="genotypes.vcf.gz",
    sample_data_path="samples.tsv",
)
print(predictions)  # DataFrame: sampleID, x, y, <covariates...>
```

## Leave-one-out

`fit_predict_loo()` trains and predicts across folds, holding samples out in
turn:

```python
model = EcoLocator()
loo = model.fit_predict_loo(
    genotype_path="genotypes.vcf.gz",
    sample_data_path="samples.tsv",
)
```

See {doc}`api/index` for the full parameter list of each method.
