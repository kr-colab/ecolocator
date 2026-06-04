# ecoLocator

`ecoLocator` is a supervised machine-learning method that predicts the
geographic origin **and** environmental covariates of a sample from genotype
data. It extends [locator](https://github.com/kr-colab/locator)
([Battey et al., eLife 2020](https://elifesciences.org/articles/54507)), which
predicted location only.

The core idea: train a neural network on samples with known coordinates, then
predict coordinates -- and environmental covariates -- for held-out samples. The
network has a shared trunk feeding two output heads, one for geographic location
`(x, y)` and one for environmental covariates.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
cli
api/index
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
