import sys
import allel
import zarr
import numpy as np
import pandas as pd
import logging


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
        positions = callset["variants/POS"]
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