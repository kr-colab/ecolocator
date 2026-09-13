# EcoLocator

`EcoLocator` is a supervised machine-learning method that predicts the
geographic origin **and** environmental covariates of a sample from genotype
data. It extends [locator](https://github.com/kr-colab/locator)
([Battey et al., eLife 2020](https://elifesciences.org/articles/54507)), which
predicted location only.

The core idea: train a neural network on samples with known coordinates, then
predict coordinates -- and environmental covariates -- for held-out samples. The
network has a shared trunk feeding two output heads, one for geographic location
`(x, y)` and one for environmental covariates.

## At a glance

```python
from ecolocator import EcoLocator

model = EcoLocator()
model.fit("./data/test_genotypes.vcf.gz", "./data/test_sample_data_cov.txt")
predictions = model.predict("./data/test_genotypes.vcf.gz", "./data/test_sample_data_cov.txt")
```

See {doc}`getting-started-python` for the full walkthrough, or {doc}`getting-started-cli`
for the command-line equivalent.

::::{grid} 2
:gutter: 3
:margin: 2 0 0 0

:::{grid-item-card} {octicon}`desktop-download` Installation
:link: installation
:link-type: doc

Set EcoLocator up with `uv`.
:::

:::{grid-item-card} {octicon}`terminal` Command-line usage
:link: getting-started-cli
:link-type: doc

Train, predict, and attribute from the terminal.
:::

:::{grid-item-card} {octicon}`code` Python usage
:link: getting-started-python
:link-type: doc

Fit, predict, and explore results interactively.
:::

:::{grid-item-card} {octicon}`graph` Example: Douglas-fir
:link: examples/dougfir_demo
:link-type: doc

A full worked case study with real results.
:::

:::{grid-item-card} {octicon}`book` API reference
:link: api/index
:link-type: doc

Full parameter reference for every method.
:::

::::

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

installation
getting-started-cli
getting-started-python
examples/dougfir_demo
api/index
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
