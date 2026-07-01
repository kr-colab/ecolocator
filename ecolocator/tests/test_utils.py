import pytest
import numpy as np
import allel
import pandas as pd
import logging
from ecolocator.utils import (
    load_genotypes,
    sort_samples,
    replace_missing_data,
    filter_snps,
    normalize_locs,
    back_transform_env,
    _resolve_snp_ids,
)


def test_load_genos_noinput():
    """testing load genotypes errors with no inputs"""
    with pytest.raises(ValueError, match="provide one of"):
        load_genotypes()


def test_load_genos_equivalence(example_data):
    """testing load genotypes returns same across data types"""
    vcf_path, zarr_path, matrix_path, _ = example_data
    genos_vcf, samples_vcf, snp_ids_vcf = load_genotypes(vcf_path=str(vcf_path))
    genos_zarr, samples_zarr, snp_ids_zarr = load_genotypes(zarr_path=str(zarr_path))
    genos_matrix, samples_matrix, snp_ids_matrix = load_genotypes(
        matrix_path=str(matrix_path)
    )

    # VCF and zarr are both phased — arrays should be exactly equal
    np.testing.assert_array_equal(genos_vcf, genos_zarr)
    np.testing.assert_array_equal(samples_vcf, samples_zarr)

    # Matrix stores dosage only (phase is lost) — compare alt allele counts
    np.testing.assert_array_equal(genos_vcf.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(genos_zarr.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(samples_vcf, samples_matrix)
    np.testing.assert_array_equal(samples_zarr, samples_matrix)
    np.testing.assert_array_equal(snp_ids_vcf, snp_ids_zarr)
    # matrix SNP ids should be exactly the file's own column headers
    expected_matrix_ids = pd.read_csv(matrix_path, sep="\t", nrows=0).columns.drop(
        "sampleID"
    )
    np.testing.assert_array_equal(snp_ids_matrix, np.array(expected_matrix_ids))


def test_sort_samples(example_data):
    """testing sort samples returns the expected output with correctly formatted input"""
    vcf_path, _, _, sample_data_path = example_data
    genos_vcf, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path)
    assert sample_data.shape[0] == len(samples_vcf)
    np.testing.assert_array_equal(sample_data.index, samples_vcf)
    assert locs.shape == (len(samples_vcf), 5)
    assert not np.any(np.isnan(locs))


def test_resolve_snp_ids_all_present():
    """testing _resolve_snp_ids returns real IDs unchanged when none are missing"""
    ids = np.array(["rs1", "rs2", "rs3"])
    chrom = np.array(["1", "1", "2"])
    pos = np.array([100, 200, 300])
    result = _resolve_snp_ids(ids, chrom, pos)
    np.testing.assert_array_equal(result, ids)


def test_resolve_snp_ids_any_missing_falls_back_for_all(caplog):
    """testing _resolve_snp_ids falls back to CHROM:POS for every SNP if any ID is missing"""
    ids = np.array(["rs1", ".", "rs3"])
    chrom = np.array(["1", "1", "2"])
    pos = np.array([100, 200, 300])
    with caplog.at_level(logging.WARNING):
        result = _resolve_snp_ids(ids, chrom, pos)
    np.testing.assert_array_equal(result, ["1:100", "1:200", "2:300"])
    assert any("1/3" in message for message in caplog.messages)


def test_sort_samples_errors(example_data, tmp_path):
    """testing sort samples raises errors when samples are missing, etc"""
    vcf_path, _, _, _ = example_data
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))

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
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path, covariates=[])
    assert locs.shape == (len(samples_vcf), 2)
    assert not np.any(np.isnan(locs))


def test_sort_samples_specific_covariates(example_data):
    """testing sort_samples respects an explicit covariate list"""
    vcf_path, _, _, sample_data_path = example_data
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))
    sample_data, locs = sort_samples(samples_vcf, sample_data_path, covariates=["cov1"])
    assert locs.shape == (len(samples_vcf), 3)
    assert not np.any(np.isnan(locs))


def test_sort_samples_count_mismatch(example_data, tmp_path):
    """testing sort_samples raises an informative error when sample counts differ but names match"""
    vcf_path, _, _, _ = example_data
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))

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
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))

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
    _, samples_vcf, _ = load_genotypes(vcf_path=str(vcf_path))
    with pytest.raises(ValueError, match="not found in sample data"):
        sort_samples(samples_vcf, sample_data_path, covariates=["nonexistent_col"])


# Testing no missing data -> output should equal input's alt count
def test_replace_missing_data_no_missing(example_data):
    """replace_missing_data on fully-observed data returns same alt counts"""
    vcf_path, _, _, _ = example_data
    genos, _, _ = load_genotypes(vcf_path=str(vcf_path))
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


def test_filter_snps_returns_correct_types(example_data):
    """testing filter_snps returns a GenotypeArray and a numpy array of indices"""
    vcf_path, _, _, _ = example_data
    genos, _, _ = load_genotypes(vcf_path=str(vcf_path))
    filtered, kept_indices = filter_snps(genos)
    assert isinstance(filtered, allel.GenotypeArray)
    assert isinstance(kept_indices, np.ndarray)


def test_filter_snps_biallelic_only():
    """testing filter_snps removes monomorphic SNPs, keeping only biallelic ones"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [0, 0], [1, 1]],  # biallelic - kept
            [[0, 0], [0, 0], [0, 0], [0, 0]],  # monomorphic - removed
            [[0, 1], [1, 1], [0, 0], [0, 1]],  # biallelic - kept
        ]
    )
    filtered, kept_indices = filter_snps(gt, min_mac=1)
    assert len(filtered) == 2
    np.testing.assert_array_equal(kept_indices, [0, 2])


def test_filter_snps_mac_filter():
    """testing filter_snps removes SNPs whose alt allele count falls below min_mac"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [0, 0], [0, 0]],  # mac=1 - removed with min_mac=2
            [[0, 1], [0, 0], [0, 0], [0, 1]],  # mac=2 - kept
            [[1, 1], [0, 1], [0, 1], [0, 0]],  # mac=4 - kept
        ]
    )
    filtered, kept_indices = filter_snps(gt, min_mac=2)
    assert len(filtered) == 2
    np.testing.assert_array_equal(kept_indices, [1, 2])


def test_filter_snps_min_mac_1_skips_mac_filter():
    """testing filter_snps with min_mac=1 skips MAC filtering, keeping all biallelic SNPs"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [0, 0], [0, 0]],  # mac=1
            [[0, 1], [0, 0], [0, 0], [0, 1]],  # mac=2
        ]
    )
    filtered_mac1, _ = filter_snps(gt, min_mac=1)
    filtered_mac2, _ = filter_snps(gt, min_mac=2)
    assert len(filtered_mac1) == 2
    assert len(filtered_mac2) == 1


def test_filter_snps_max_snps():
    """testing filter_snps with max_snps returns exactly that many SNPs"""
    gt = allel.GenotypeArray([[[0, 1], [0, 1], [0, 0], [0, 0]] for _ in range(10)])
    filtered, kept_indices = filter_snps(gt, max_snps=3, rng=np.random.default_rng(42))
    assert len(filtered) == 3
    assert len(kept_indices) == 3


def test_filter_snps_rng_reproducible():
    """filter_snps with the same seeded rng selects the same SNPs each time"""
    gt = allel.GenotypeArray([[[0, 1], [0, 1], [0, 0], [0, 0]] for _ in range(10)])
    filtered1, indices1 = filter_snps(gt, max_snps=3, rng=np.random.default_rng(42))
    filtered2, indices2 = filter_snps(gt, max_snps=3, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_equal(filtered1, filtered2)


def test_filter_snps_kept_indices_consistent():
    """testing if kept_indices correctly references the matching rows in the original genotypes array"""
    gt = allel.GenotypeArray(
        [
            [[0, 0], [0, 1], [0, 0], [0, 0]],  # mac=1 - removed
            [[0, 1], [0, 0], [0, 0], [0, 1]],  # mac=2 - kept
            [[1, 1], [0, 1], [0, 1], [0, 0]],  # mac=4 - kept
        ]
    )
    filtered, kept_indices = filter_snps(gt, min_mac=2)
    np.testing.assert_array_equal(filtered, gt[kept_indices])


def test_normalize_locs_zscore_properties(sample_locs):
    """testing normalize_locs with default transforms produces mean=0, std=1 for all columns"""
    *_, norm_locs = normalize_locs(sample_locs)
    np.testing.assert_allclose(np.mean(norm_locs, axis=0), np.zeros(5), atol=1e-10)
    np.testing.assert_allclose(np.std(norm_locs, axis=0), np.ones(5), atol=1e-10)


def test_normalize_locs_default_transforms_equivalent(sample_locs):
    """testing normalize_locs with transform=None gives same result as explicit all-none transforms"""
    result_default = normalize_locs(sample_locs)
    result_explicit = normalize_locs(sample_locs, ["none", "none", "none"])
    np.testing.assert_array_equal(result_default[-1], result_explicit[-1])
    assert result_default[6] == result_explicit[6]


def test_normalize_locs_roundtrip_none(sample_locs):
    """testing normalize_locs with none transform recovers original values on back-transform"""
    meanlong, sdlong, meanlat, sdlat, means, sds, _, norm_locs = normalize_locs(
        sample_locs
    )
    recovered = np.column_stack(
        [
            norm_locs[:, 0] * sdlong + meanlong,
            norm_locs[:, 1] * sdlat + meanlat,
            norm_locs[:, 2:] * sds + means,
        ]
    )
    np.testing.assert_allclose(recovered, sample_locs, atol=1e-10)


def test_normalize_locs_log_is_invertible(sample_locs):
    """testing normalize_locs with log transform recovers original covariate values on back-transform"""
    meanlong, sdlong, meanlat, sdlat, means, sds, _, norm_locs = normalize_locs(
        sample_locs, transforms=["none", "none", "log"]
    )
    recovered_covs = np.column_stack(
        [
            norm_locs[:, 2] * sds[0] + means[0],  # cov1 — none
            norm_locs[:, 3] * sds[1] + means[1],  # cov2 — none
            np.exp(norm_locs[:, 4] * sds[2] + means[2]),  # cov3 — log
        ]
    )
    np.testing.assert_allclose(recovered_covs, sample_locs[:, 2:], atol=1e-10)


def test_normalize_locs_transforms_length_mismatch():
    """testing normalize_locs raises ValueError when transforms length does not match number of covariates"""
    locs = np.array(
        [
            [1.0, 2.0, 10.0, 100.0, 5.0],
            [2.0, 3.0, 20.0, 200.0, 10.0],
            [3.0, 4.0, 30.0, 300.0, 15.0],
        ]
    )
    with pytest.raises(ValueError, match="transforms length"):
        normalize_locs(locs, transforms=["none", "none"])


def test_normalize_locs_unknown_transform(sample_locs):
    """testing normalize_locs raises ValueError when an unrecognized transform name is passed"""
    with pytest.raises(ValueError, match="Unknown transform"):
        normalize_locs(sample_locs, transforms=["none", "sqrt", "none"])


def test_normalize_locs_log_nonpositive_raises():
    """testing that normalize_locs raises ValueError when log transform is applied to a covariate with non-positive values"""
    locs = np.array(
        [
            [1.0, 2.0, -5.0, 100.0, 5.0],
            [2.0, 3.0, 20.0, 200.0, 10.0],
            [3.0, 4.0, 30.0, 300.0, 15.0],
        ]
    )
    with pytest.raises(ValueError, match="non-positive values"):
        normalize_locs(locs, transforms=["log", "none", "none"])


def test_normalize_locs_log_zero_raises():
    """testing that normalize_locs raises ValueError when log transform is applied to a covariate containing zero"""
    locs = np.array(
        [
            [1.0, 2.0, 0.0, 100.0, 5.0],
            [2.0, 3.0, 20.0, 200.0, 10.0],
            [3.0, 4.0, 30.0, 300.0, 15.0],
        ]
    )
    with pytest.raises(ValueError, match="non-positive values"):
        normalize_locs(locs, transforms=["log", "none", "none"])


def test_normalize_locs_warns_small_gap(caplog):
    """testing that normalize_locs emits warning when a pos. covariate has a small gap to the zero boundary"""
    locs = np.array(
        [
            [1.0, 2.0, 22.0, 100.0, 5.0],
            [2.0, 3.0, 23.0, 200.0, 10.0],
            [3.0, 4.0, 24.0, 300.0, 15.0],
            [4.0, 5.0, 25.0, 400.0, 20.0],
            [5.0, 6.0, 98.0, 500.0, 25.0],
        ]
    )
    with caplog.at_level(logging.WARNING):
        normalize_locs(locs, transforms=["none", "none", "none"])
    assert any("cov1" in message for message in caplog.messages)


def test_normalize_locs_no_warning_with_log(caplog):
    """testing that normalize_locs does not warn when log transform is specified for a skewed positive covariate"""
    locs = np.array(
        [
            [1.0, 2.0, 5.0, 1000.0, 500.0],
            [2.0, 3.0, 6.0, 1100.0, 550.0],
            [3.0, 4.0, 7.0, 1200.0, 600.0],
            [4.0, 5.0, 8.0, 1300.0, 650.0],
            [5.0, 6.0, 98.0, 1400.0, 700.0],
        ]
    )
    with caplog.at_level(logging.WARNING):
        normalize_locs(locs, transforms=["log", "none", "none"])
    assert len(caplog.messages) == 0


def test_normalize_locs_input_not_mutated(sample_locs):
    """testing that normalize_locs does not modify the input locs array"""
    original = sample_locs.copy()
    normalize_locs(sample_locs, transforms=["none", "log", "none"])
    np.testing.assert_array_equal(sample_locs, original)


def test_normalize_locs_nan_prediction_samples():
    """testing that normalize_locs handles prediction samples with NaN in all columns"""
    locs = np.array(
        [
            [1.0, 2.0, 10.0, 100.0, 5.0],
            [2.0, 3.0, 20.0, 200.0, 10.0],
            [3.0, 4.0, 30.0, 300.0, 15.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    *_, norm_locs = normalize_locs(locs)
    assert np.all(np.isnan(norm_locs[3:]))
    assert np.all(~np.isnan(norm_locs[:3]))


def test_back_transform_env_none_is_linear():
    """testing back_tranform_env with all-none transforms is exactly z * sds + means"""
    z = np.array([[0.0, 1.0, -1.0], [1.0, -1.0, 0.0]])
    means = np.array([10.0, 20.0, 30.0])
    sds = np.array([2.0, 4.0, 5.0])
    expected = np.array([[10.0, 24.0, 25.0], [12.0, 16.0, 30.0]])
    result = back_transform_env(z, means, sds, ["none", "none", "none"])
    np.testing.assert_array_equal(result, expected)


def test_back_transform_env_log_always_positive():
    """testing back_transform_env with log transform always returns positive values, even for extreme negative z"""
    z = np.array([[-100.0], [-10.0], [0.0]])
    means = np.array([5.0])
    sds = np.array([2.0])
    result = back_transform_env(z, means, sds, ["log"])
    assert np.all(result > 0)


def test_back_transform_env_mixed():
    """testing back_transform_env applies none and log transforms to the correct columns independently"""
    z = np.array([[1.0, 2.0], [0.0, 1.0]])
    means = np.array([10.0, 3.0])
    sds = np.array([2.0, 1.0])
    result = back_transform_env(z, means, sds, ["none", "log"])
    # col 0: linear  — 1.0*2.0+10.0=12.0, 0.0*2.0+10.0=10.0
    # col 1: log     — exp(2.0*1.0+3.0)=exp(5.0), exp(1.0*1.0+3.0)=exp(4.0)
    expected = np.array([[12.0, np.exp(5.0)], [10.0, np.exp(4.0)]])
    np.testing.assert_allclose(result, expected)


def test_back_transform_env_roundtrip(sample_locs):
    """testing back_transform_env composed with normalize_locs recovers original covariate values"""
    _, _, _, _, means, sds, transforms, norm_locs = normalize_locs(
        sample_locs, transforms=["none", "log", "none"]
    )
    recovered = back_transform_env(norm_locs[:, 2:], means, sds, transforms)
    np.testing.assert_allclose(recovered, sample_locs[:, 2:], atol=1e-10)


def test_back_transform_env_output_shape():
    """back_transform_env output shape matches input shape"""
    z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    means = np.array([1.0, 2.0, 3.0])
    sds = np.array([1.0, 1.0, 1.0])
    result = back_transform_env(z, means, sds, ["none", "log", "none"])
    assert result.shape == z.shape
