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
def predict(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        file_okay=False, 
        help="Directory containing a saved EcoLocator model (from `ecolocator train`).",
    ),
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
        help="Sample metadata TSV. Rows with missing x/y are predicted.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for predictions.",
    ),
) -> None:
    """Predict locations and covariates for samples with unknown coordinates."""
    from .ecolocator_class import EcoLocator

    el = EcoLocator.load(str(model))
    result = el.predict(str(genotypes), str(sample_data))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, sep="\t", index=False)
    typer.echo(f"Predicted {len(result)} samples → {out}")


@app.command()
def loo(
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
        help="Sample metadata TSV with sampleID, x, y, and covariate columns (all rows known).",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for combined LOO predictions.",
    ),
    max_folds: Optional[int] = typer.Option(
        None,
        help="Stop after this many folds. Useful for smoke-testing before a full run.",
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
    verbose: int = typer.Option(0, help="Keras verbosity (0=silent, 1=batches, 2=epochs)."),
) -> None:
    """Leave-one-out fit-and-predict over all samples with known coordinates."""
    from .ecolocator_class import EcoLocator

    transforms = [t.strip() for t in cov_transforms.split(",")] if cov_transforms else None

    el = EcoLocator(
        cov_transforms=transforms,
        nlayers=nlayers,
        width=width,
        dropout_prop=dropout_prop,
        loc_weight=loc_weight,
        env_weight=env_weight,
    )
    result = el.fit_predict_loo(
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
        max_folds=max_folds,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, sep="\t", index=False)
    typer.echo(f"LOO complete: {len(result)} predictions → {out}")


@app.command()
def attribute(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        file_okay=False,
        help="Directory containing a saved EcoLocator model (from `ecolocator train`).",
    ),
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
        help="Sample metadata TSV. Rows with missing x/y are attributed; remaining rows form the background.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output TSV path for mean absolute SHAP summary (rows=SNPs, cols=output variables).",
    ),
    background_size: int = typer.Option(
        100,
        help="Number of training samples to use as SHAP background. Larger values are slower but more stable.",
    ),
    min_maf: Optional[float] = typer.Option(
        None,
        help="Exclude SNPs below this minor allele frequency from the output.",
    ),
    save_raw: bool = typer.Option(
        False,
        "--save-raw",
        help="Also save the per-sample raw SHAP values to <out-stem>_raw.tsv.",
    ),
    seed: Optional[int] = typer.Option(None, help="RNG seed for background sampling."),
) -> None:
    """Attribute SNP importance for samples with unknown coordinates using SHAP."""
    from .ecolocator_class import EcoLocator

    el = EcoLocator.load(str(model))
    out.parent.mkdir(parents=True, exist_ok=True)

    result = el.shap_values(
        genotype_path=str(genotypes),
        sample_data_path=str(sample_data),
        train_genotype_path=str(genotypes),
        train_sample_data_path=str(sample_data),
        background_size=background_size,
        min_maf=min_maf,
        seed=seed,
        raw=False,
    )
    result.to_csv(out, sep="\t", index=False)
    typer.echo(f"Attributed {len(result)} SNPs → {out}")

    if save_raw:
        raw_result = el.shap_values(
            genotype_path=str(genotypes),
            sample_data_path=str(sample_data),
            train_genotype_path=str(genotypes),
            train_sample_data_path=str(sample_data),
            background_size=background_size,
            min_maf=min_maf,
            seed=seed,
            raw=True,
        )
        raw_out = out.with_stem(out.stem + "_raw")
        raw_result.to_csv(raw_out, sep="\t", index=False)
        typer.echo(f"Raw SHAP values → {raw_out}")


if __name__ == "__main__":
    app()
