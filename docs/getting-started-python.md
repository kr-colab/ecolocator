# Python usage

This page shows the Python API. For the equivalent command-line workflow, see
{doc}`getting-started-cli`.

## Inputs


## Train a model

```python
from ecolocator import EcoLocator

model = EcoLocator()
model.fit(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
model.save("my_model")
```

`save()` writes a directory containing `model.keras`, `arrays.npz`, and
`params.json`.

## Predict

```python
from ecolocator import EcoLocator

model = EcoLocator.load("my_model")
predictions = model.predict(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
print(predictions)  # DataFrame: sampleID, x, y, <covariates...>
```


## Leave-one-out

`fit_predict_loo()` trains and predicts across folds, holding samples out in
turn:

```python
model = EcoLocator()
loo = model.fit_predict_loo(
    genotype_path="./data/test_genotypes.vcf.gz",
    sample_data_path="./data/test_sample_data_cov.txt",
)
```

## SHAP feature attribution

```python
shap_out = model.shap_values(genotype_path="./data/test_genotypes.vcf.gz",
                  sample_data_path="./data/test_sample_data_cov.txt",
                  train_genotype_path ="./data/test_genotypes.vcf.gz",
                  train_sample_data_path="./data/test_sample_data_cov.txt"
                  )
```

See {doc}`api/index` for the full parameter list of each method.
