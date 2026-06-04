# Command-line interface

Installing the package provides the `ecolocator` command. Every subcommand has
`--help`:

```bash
ecolocator --help
ecolocator train --help
```

## `ecolocator train`

Train a model and save it to a directory.

```bash
ecolocator train \
    --genotypes genotypes.vcf.gz \
    --sample-data samples.tsv \
    --out my_model
```

Key options mirror the {class}`~ecolocator.ecolocator_class.EcoLocator`
constructor and `fit()` method -- network shape (`--nlayers`, `--width`,
`--dropout-prop`), loss weights (`--loc-weight`, `--env-weight`), SNP filtering
(`--min-mac`, `--max-snps`), and training control (`--max-epochs`, `--patience`,
`--batch-size`, `--train-split`, `--seed`). See `ecolocator train --help` for the
full list.

## `ecolocator predict`

Predict locations and covariates for samples with unknown (`NA`) coordinates
using a saved model.

```bash
ecolocator predict --help
```

## `ecolocator loo`

Leave-one-out fit-and-predict over the samples with known coordinates.

```bash
ecolocator loo --help
```
