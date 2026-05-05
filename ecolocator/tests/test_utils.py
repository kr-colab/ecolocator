import pytest
import numpy as np
import msprime
import allel
import pandas as pd
from ecolocator.utils import load_genotypes, sort_samples, replace_missing_data


@pytest.fixture(scope="session")
def example_data(tmp_path_factory):
    num_covariates = 3
    num_individuals = 5
    random_seed = 1024
    # use msprime to simulate some data
    ts = msprime.sim_ancestry(
        num_individuals,
        sequence_length=1e4,
        population_size=1e4,
        random_seed=random_seed,
        recombination_rate=1e-8,
    )
    ts = msprime.sim_mutations(
        ts, rate=1e-8, model=msprime.BinaryMutationModel(), random_seed=random_seed
    )
    # paths to temp files
    vcf_path = tmp_path_factory.mktemp("data") / "test.vcf"
    zarr_path = tmp_path_factory.mktemp("data") / "test.zarr"
    matrix_path = tmp_path_factory.mktemp("data") / "test_matrix.tsv"
    sample_data_path = tmp_path_factory.mktemp("data") / "test_sample_data.tsv"
    # write data to the temp files
    sample_ids = [f"sample{i}" for i in range(ts.num_individuals)]
    with open(vcf_path, "w") as vcf_file:
        ts.write_vcf(vcf_file, position_transform="legacy", individual_names=sample_ids)
    allel.vcf_to_zarr(str(vcf_path), str(zarr_path), fields="*", overwrite=True)
    haplo_matrix = ts.genotype_matrix().T
    diplo_matrix = haplo_matrix[::2] + haplo_matrix[1::2]
    df = pd.DataFrame(diplo_matrix, index=sample_ids)
    df.index.name = "sampleID"
    df.to_csv(matrix_path, sep="\t", index=True)

    rng = np.random.default_rng(random_seed)
    locs = rng.uniform(0, 1, size=(ts.num_individuals, 2 + num_covariates))
    shuffled_indices = rng.permutation(ts.num_individuals)
    col_names = ["x", "y"] + [f"cov{i + 1}" for i in range(num_covariates)]
    sample_data_df = pd.DataFrame(
        locs[shuffled_indices],
        index=[f"sample{i}" for i in shuffled_indices],
        columns=col_names,
    )
    sample_data_df.index.name = "sampleID"
    sample_data_df.to_csv(sample_data_path, sep="\t", index=True)

    # TODO:
    # .  you could also have a second fixture that returns the same genotype data but with more/fewer smaples, to test errors in sort_samples
    return (vcf_path, zarr_path, matrix_path, sample_data_path)


def test_load_genos_noinput():
    """testing load genotypes errors with no inputs"""
    with pytest.raises(ValueError, match="provide one of"):
        load_genotypes()


def test_load_genos_equivalence(example_data):
    """testing load genotypes returns same across data types"""
    vcf_path, zarr_path, matrix_path, _ = example_data
    genos_vcf, samples_vcf = load_genotypes(vcf_path=str(vcf_path))
    genos_zarr, samples_zarr = load_genotypes(zarr_path=str(zarr_path))
    genos_matrix, samples_matrix = load_genotypes(matrix_path=str(matrix_path))

    # VCF and zarr are both phased — arrays should be exactly equal
    np.testing.assert_array_equal(genos_vcf, genos_zarr)
    np.testing.assert_array_equal(samples_vcf, samples_zarr)

    # Matrix stores dosage only (phase is lost) — compare alt allele counts
    np.testing.assert_array_equal(genos_vcf.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(genos_zarr.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(samples_vcf, samples_matrix)
    np.testing.assert_array_equal(samples_zarr, samples_matrix)


def test_sort_samples(example_data):
    """testing sort samples returns the expected output with correctly formatted input"""
    vcf_path, _, _, sample_data_path = example_data
    genos_vcf, samples_vcf = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path)
    assert sample_data.shape[0] == len(samples_vcf)
    np.testing.assert_array_equal(sample_data.index, samples_vcf)
    assert locs.shape == (len(samples_vcf), 5)
    assert not np.any(np.isnan(locs))


def test_sort_samples_errors(example_data, tmp_path):
    """testing sort samples raises errors when samples are missing, etc"""
    vcf_path, _, _, _ = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))

    # create a sample data file with missing samples
    bad_sample_ids = list(samples_vcf[:-1]) + ["missing_sample"]
    bad_df = pd.DataFrame(
        np.zeros((len(bad_sample_ids), 3)),
        index=bad_sample_ids,
        columns=["x", "y", "cov1"],
    )
    bad_df.index.name = "sampleID"
    bad_df.to_csv(tmp_path / "bad_sample_data.tsv", sep="\t", index=True)
    with pytest.raises(ValueError, match="sample ordering failed"):
        sort_samples(samples_vcf, str(tmp_path / "bad_sample_data.tsv"))


def test_sort_samples_no_covs(example_data):
    """testing sort_samples works with no covariates (location prediction only)"""
    vcf_path, _, _, sample_data_path = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path, covariates=[])
    assert locs.shape == (len(samples_vcf), 2)
    assert not np.any(np.isnan(locs))


def test_sort_samples_specific_covariates(example_data):
    """testing sort_samples respects an explicit covariate list"""
    vcf_path, _, _, sample_data_path = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path, covariates=["cov1"])
    assert locs.shape == (len(samples_vcf), 3)
    assert not np.any(np.isnan(locs))


def test_sort_samples_count_mismatch(example_data, tmp_path):
    """testing sort_samples raises an informative error when sample counts differ but names match"""
    vcf_path, _, _, _ = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))

    # sample data with only the first 4 of 5 VCF samples — all names valid, one missing
    subset_ids = list(samples_vcf[:-1])
    subset_df = pd.DataFrame(
        np.zeros((len(subset_ids), 3)), index=subset_ids, columns=["x", "y", "cov1"]
    )
    subset_df.index.name = "sampleID"
    subset_df.to_csv(tmp_path / "subset_sample_data.tsv", sep="\t", index=True)
    with pytest.raises(ValueError, match="counts differ"):
        sort_samples(samples_vcf, str(tmp_path / "subset_sample_data.tsv"))


def test_sort_samples_count_mismatch_more(example_data, tmp_path):
    """testing sort_samples raises an informative error when TSV has more rows than the VCF"""
    vcf_path, _, _, _ = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))

    # all 5 VCF samples plus one extra
    extra_ids = list(samples_vcf) + ["extra_sample"]
    extra_df = pd.DataFrame(
        np.zeros((len(extra_ids), 3)), index=extra_ids, columns=["x", "y", "cov1"]
    )
    extra_df.index.name = "sampleID"
    extra_df.to_csv(tmp_path / "extra_sample_data.tsv", sep="\t", index=True)
    with pytest.raises(ValueError, match="counts differ"):
        sort_samples(samples_vcf, str(tmp_path / "extra_sample_data.tsv"))


def test_sort_samples_invalid_covariate(example_data):
    """testing sort_samples raises a clear error when a requested covariate column does not exist"""
    vcf_path, _, _, sample_data_path = example_data
    _, samples_vcf = load_genotypes(vcf_path=str(vcf_path))
    with pytest.raises(ValueError, match="not found in sample data"):
        sort_samples(samples_vcf, sample_data_path, covariates=["nonexistent_col"])


# Testing no missing data -> output should equal input's alt count
def test_replace_missing_data_no_missing(example_data):
    """replace_missing_data on fully-observed data returns same alt counts"""
    vcf_path, _, _, _ = example_data
    genos, _ = load_genotypes(vcf_path=str(vcf_path))
    expected = genos.to_allele_counts()[:, :, 1]
    result = replace_missing_data(genos)
    np.testing.assert_array_equal(result, expected)


# missing calls are filled, valid values only
def test_replace_missing_data_fills_missing():
    """replace_missing_data fills all missing calls with values in {0, 1, 2}"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [-1, -1], [1, 1]],
            [[0, 1], [-1, -1], [0, 1], [0, 0]],
            [[-1, -1], [1, 1], [0, 1], [0, 0]],
        ]
    )
    result = replace_missing_data(gt)
    assert result.shape == (3, 4)
    assert np.all(np.isin(result, [0, 1, 2]))


# test that non-missing positions are not touched
def test_replace_missing_data_non_missing_unchanged():
    """replace_missing_data does not alter observed (non-missing) genotype counts"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [-1, -1], [1, 1]],
            [[0, 1], [0, 0], [-1, -1], [0, 0]],
        ]
    )
    expected = gt.to_allele_counts()[:, :, 1]
    result = replace_missing_data(gt)
    # samples 0, 1, 3 have no missing data — counts must be unchanged
    np.testing.assert_array_equal(result[:, 0], expected[:, 0])
    np.testing.assert_array_equal(result[:, 1], expected[:, 1])
    np.testing.assert_array_equal(result[:, 3], expected[:, 3])


print("hello")