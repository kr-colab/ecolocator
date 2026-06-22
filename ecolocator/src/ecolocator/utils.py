import sys
import allel
import zarr
import numpy as np
import pandas as pd
import logging
from scipy import stats


def load_genotypes(
    zarr_path: str = None,
    vcf_path: str = None,
    matrix_path: str = None,
) -> (allel.GenotypeArray, np.ndarray):
    if zarr_path is not None:
        logging.info("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        _positions = callset["variants/POS"]
    elif vcf_path is not None:
        logging.info("reading VCF")
        vcf = allel.read_vcf(vcf_path, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif matrix_path is not None:
        gmat = pd.read_csv(matrix_path, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")
        for i in range(
            gmat.shape[0]
        ):  # kludge to get haplotypes for reading in to allel.
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))
        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
    else:
        raise ValueError("Must provide one of zarr_path, vcf_path, or matrix_path")
    logging.info(f"loaded {np.shape(genotypes)} genotypes")
    logging.info(f"loaded {len(samples)} samples")
    return genotypes, samples


def sort_samples(
    samples: np.ndarray,
    sample_data_path: str,
    covariates: list = None,
) -> (pd.DataFrame, np.ndarray):
    """
    `sample_data_path` points to a tsv with required columns "sampleID", "x", "y".
    Any additional columns are treated as covariates unless `covariates` is given.

    Parameters
    ----------
    covariates : list of str or None
        Column names to use as covariates. None (default) uses all columns
        that are not "sampleID", "x", or "y". Pass [] for no covariates.
    """
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    if len(sample_data) != len(samples):
        vcf_set = set(samples)
        data_set = set(sample_data.index)
        if data_set <= vcf_set or vcf_set <= data_set:
            raise ValueError(
                f"Sample names match but counts differ: VCF has {len(samples)} "
                f"samples, sample data has {len(sample_data)} rows."
            )
    sample_data = sample_data.reindex(
        np.array(samples)
    )  # sort loc table so samples are in same order as vcf samples
    if not all(
        [sample_data["sampleID2"].iloc[x] == samples[x] for x in range(len(samples))]
    ):  # check that all sample names are present
        raise ValueError("sample ordering failed! Check that sample IDs match the VCF.")
    if covariates is None:
        reserved = {"sampleID2", "x", "y"}
        cov_cols = [c for c in sample_data.columns if c not in reserved]
    else:
        missing = [c for c in covariates if c not in sample_data.columns]
        if missing:
            raise ValueError(
                f"Covariate column(s) not found in sample data: {missing}. "
                f"Available columns: {list(sample_data.columns)}."
            )
        cov_cols = list(covariates)
    locs = np.array(sample_data[["x", "y"] + cov_cols])
    logging.debug(f"Input data:\n{locs}")
    logging.info(f"sorted samples and covariates for {len(samples)} samples")
    return sample_data, locs


def replace_missing_data(
    genotypes: allel.GenotypeArray, rng: np.random.Generator = None
) -> np.ndarray:
    logging.info("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = dc / (2 * ninds)
    for i in range(np.shape(ac)[0]):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                if rng is not None:
                    ac[i, j] = rng.binomial(2, af[i])
                else:
                    ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(
    genotypes: allel.GenotypeArray,
    min_mac: int = 2,
    max_snps: int = None,
    rng: np.random.Generator = None,
) -> tuple:
    logging.info("filtering SNPs...")
    kept_indices = np.arange(genotypes.shape[0])
    allele_counts = genotypes.count_alleles()
    biallelic_mask = allele_counts.is_biallelic()
    genotypes = genotypes[biallelic_mask, :, :]
    kept_indices = kept_indices[biallelic_mask]
    if min_mac != 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        mac_mask = derived_counts >= min_mac
        genotypes = genotypes[mac_mask, :, :]
        kept_indices = kept_indices[mac_mask]
    if max_snps is not None:
        if rng is None:
            rng = np.random.default_rng()
        selected = rng.choice(len(genotypes), max_snps, replace=False)
        genotypes = genotypes[selected, :, :]
        kept_indices = kept_indices[selected]
    logging.info(f"Retained {len(genotypes)} SNPs after filtering")
    return genotypes, kept_indices


def normalize_locs(
    locs: np.array,
    transforms: list = None,
    cov_names: list = None,
) -> tuple:
    num_covs = locs.shape[1] - 2
    if transforms is None:
        transforms = ["none"] * num_covs
    if cov_names is None:
        cov_names = [f"cov{i + 1}" for i in range(num_covs)]
    if len(transforms) != num_covs:
        raise ValueError(
            f"transforms length({len(transforms)}) must be equal number of "
            f"covariates ({num_covs})"
        )
    valid = {"none", "log"}
    unknown = [t for t in transforms if t not in valid]
    if unknown:
        raise ValueError(
            f"Unknown transform(s): {unknown}. Valid options: {sorted(valid)}."
        )
    for i, t in enumerate(transforms):
        if t == "log":
            col = locs[:, 2 + i]
            nonnan = col[~np.isnan(col)]
            if np.any(nonnan <= 0):
                raise ValueError(
                    f"{cov_names[i]} contains non-positive values "
                    f"(min={nonnan.min():.4f}); log transform requires all values > 0."
                )
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])

    cov_data = locs[:, 2:].copy().astype(float)
    for i, t in enumerate(transforms):
        if t == "log":
            cov_data[:, i] = np.log(cov_data[:, i])

    means = np.nanmean(cov_data, axis=0)
    sds = np.nanstd(cov_data, axis=0)

    for i, t in enumerate(transforms):
        if t == "none":
            raw_col = locs[:, 2 + i]
            nonnan = raw_col[~np.isnan(raw_col)]
            if np.all(nonnan > 0):
                gap = nonnan.min() / sds[i]
                skewness = stats.skew(nonnan)
                if gap < 1.0 and skewness > 1.0:
                    logging.warning(
                        f"{cov_names[i]} is strictly positive but the zero boundary in "
                        f"z-space is only {gap:.2f} units below the training minimum. "
                        f"Predictions may be negative. "
                        f"Consider using 'log' for this covariate."
                    )

    x_norm = (locs[:, 0] - meanlong) / sdlong
    y_norm = (locs[:, 1] - meanlat) / sdlat
    cov_norm = (cov_data - means) / sds
    norm_locs = np.column_stack([x_norm, y_norm, cov_norm])

    return meanlong, sdlong, meanlat, sdlat, means, sds, transforms, norm_locs


def back_transform_env(
    z_array: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    transforms: list,
) -> np.ndarray:
    result = z_array * sds + means
    for i, t in enumerate(transforms):
        if t == "log":
            result[:, i] = np.exp(result[:, i])
    return result
