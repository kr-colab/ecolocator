from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(
    add_completion=False,
    help="ecoLocator: predict geographic origin and environmental covariates from genotype data.",
    no_args_is_help=True,
)


@app.command()
def train(
    genotypes: Path = typer.Option(
        ...,
        "--genotypes",
        "-g",
        exists=True,
        readable=True,
        help="Genotype input: .vcf, .vcf.gz, .zarr, or dosage-matrix TSV.",
    ),
    sample_data: Path = typer.Option(
        ...,
        "--sample-data",
        "-s",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Sample metadata TSV with sampleID, x, y, and covariate columns. Use NA for x/y of samples to predict.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        file_okay=False,
        help="Directory to save the fitted model (model.keras, arrays.npz, params.json).",
    ),
    nlayers: int = typer.Option(10, help="Number of hidden layers in the trunk."),
    width: int = typer.Option(256, help="Units per hidden layer."),
    dropout_prop: float = typer.Option(0.25, help="Dropout proportion."),
    loc_weight: float = typer.Option(1.0, help="Loss weight on the location head."),
    env_weight: float = typer.Option(1.0, help="Loss weight on the environmental-covariate head."),
    cov_transforms: Optional[str] = typer.Option(
        None,
        help="Comma-separated per-covariate transforms (e.g. 'none,log,log'). Valid: none, log.",
    ),
    max_epochs: int = typer.Option(5000, help="Maximum training epochs."),
    patience: int = typer.Option(100, help="Early-stopping patience (epochs without val improvement)."),
    batch_size: int = typer.Option(32, help="Minibatch size."),
    min_mac: int = typer.Option(2, help="Minimum minor-allele count to retain a site."),
    max_snps: Optional[int] = typer.Option(None, help="Randomly subsample to this many SNPs (default: keep all)."),
    train_split: float = typer.Option(0.9, help="Fraction of known-location samples used for training."),
    seed: Optional[int] = typer.Option(None, help="RNG seed for splits and SNP subsetting."),
    verbose: int = typer.Option(1, help="Keras verbosity (0=silent, 1=batches, 2=epochs)."),
) -> None:
    """Train an EcoLocator model and save it to OUT."""
    from .ecolocator_class import EcoLocator

    transforms = [t.strip() for t in cov_transforms.split(",")] if cov_transforms else None

    model = EcoLocator(
        cov_transforms=transforms,
        nlayers=nlayers,
        width=width,
        dropout_prop=dropout_prop,
        loc_weight=loc_weight,
        env_weight=env_weight,
    )
    model.fit(
        genotype_path=str(genotypes),
        sample_data_path=str(sample_data),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        min_mac=min_mac,
        max_snps=max_snps,
        train_split=train_split,
        seed=seed,
        verbose=verbose,
    )
    model.save(str(out))
    typer.echo(f"Saved model to {out}")


@app.command()
def predict() -> None:
    """Predict locations and covariates for samples with unknown coordinates."""
    raise NotImplementedError("`ecolator predict` is not implemented yet.")


@app.command()
def loo() -> None:
    """Leave-one-out fit-and-predict over all samples with known coordinates."""
    raise NotImplementedError("`ecolator loo` is not implemented yet.")


if __name__ == "__main__":
    app()
