import numpy as np
import pandas as pd
import pytest
from ecolocator import EcoLocator


def test_fit_returns_self(example_data):
    """testing that fit() returns the EcoLocator instance for method chaining"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    result = model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
    )
    assert result is model


def test_fit_sets_attributes(example_data):
    """testing that fit() sets expected attributes on the EcoLocator instance"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    assert hasattr(model, "model_")
    assert hasattr(model, "history_")
    assert hasattr(model, "cov_names_")
    assert hasattr(model, "means_")
    assert hasattr(model, "meanlong_")
    assert hasattr(model, "_kept_snp_indices_")
    assert hasattr(model, "seed_")
    assert model.num_covs_ == 3
    assert model.cov_names_ == ["cov1", "cov2", "cov3"]


def test_predict_returns_dataframe(example_data, tmp_path):
    """testing that predict() returns a DataFrame with correct columns"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    # creating a sample data file with one masked sample
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.predict(str(matrix_path), str(masked_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y", "cov1", "cov2", "cov3"]
    assert len(result) == 1


def test_predict_positive_with_log_transform(example_data, tmp_path):
    """predict() with log transform always returns positive covariate values"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32, cov_transforms=["none", "none", "log"])
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.predict(str(matrix_path), str(masked_path))

    assert result["cov3"].iloc[0] > 0


def test_fit_predict_loo_returns_all_samples(example_data):
    """fit_predict_loo() returns one prediction row per known sample"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert "sampleID" in result.columns


def test_fit_predict_loo_max_folds(example_data):
    """fit_predict_loo() with max_folds returns exactly that many prediction rows"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        max_folds=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_save_load_roundtrip(tmp_path, example_data):
    """save() and load() round-trip restores all attributes and hyperparameters"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    save_dir = str(tmp_path / "saved_model")
    model.save(save_dir)
    loaded = EcoLocator.load(save_dir)

    assert loaded.nlayers == model.nlayers
    assert loaded.cov_names_ == model.cov_names_
    assert np.isclose(loaded.meanlong_, model.meanlong_)
    assert np.isclose(loaded.meanlat_, model.meanlat_)
    np.testing.assert_array_equal(loaded._kept_snp_indices_, model._kept_snp_indices_)
    assert loaded.seed_ == model.seed_


def test_load_predict_matches(tmp_path, example_data):
    """a model loaded from disk produces a valid prediction DataFrame"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    save_dir = str(tmp_path / "saved_model")
    model.save(save_dir)
    loaded = EcoLocator.load(save_dir)

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = loaded.predict(str(matrix_path), str(masked_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y", "cov1", "cov2", "cov3"]


def test_shap_values_not_fitted_raises(example_data):
    """testing that shap_values() raises RuntimeError if called before fit()"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    with pytest.raises(RuntimeError):
        model.shap_values(
            str(matrix_path),
            str(sample_data_path),
            str(matrix_path),
            str(sample_data_path),
        )


def test_shap_values_returns_dataframe(example_data, tmp_path):
    """testing that shap_values() returns a summary DataFrame with one row per SNP"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )

    n_snps = len(model._kept_snp_indices_)
    expected_cols = ["snp_id", "x", "y"] + model.cov_names_
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_snps
    assert list(result.columns) == expected_cols
    assert set(result["snp_id"]) == {str(i) for i in model._kept_snp_indices_}


def test_shap_values_min_maf_filters_columns(example_data, tmp_path):
    """testing that shap_values() with min_maf filters out low-frequency SNPs"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result_unfiltered = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )
    result_filtered = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        min_maf=0.4,
    )

    assert len(result_filtered) < len(result_unfiltered)


def test_fit_no_covariates(no_covariate_data):
    """fit() works with x and y only — no covariate columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    assert model.num_covs_ == 0
    assert model.cov_names_ == []


def test_predict_no_covariates(no_covariate_data, tmp_path):
    """predict() with no covariates returns only sampleID, x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.predict(str(matrix_path), str(masked_path))
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y"]


def test_fit_predict_loo_no_covariates(no_covariate_data):
    """fit_predict_loo() works with no covariates and returns only x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(nlayers=2, width=32)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        max_folds=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["sampleID", "x", "y"]


def test_shap_values_no_covariates(no_covariate_data, tmp_path):
    """shap_values() with no covariates returns only snp_id, x, y columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["snp_id", "x", "y"]


def test_shap_values_raw_returns_wide_format(example_data, tmp_path):
    """shap_values() with raw=True returns one row per unknown sample with all SNP×variable columns"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )

    n_snps = len(model._kept_snp_indices_)
    expected_n_cols = 1 + n_snps * (2 + len(model.cov_names_))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "sampleID" in result.columns
    assert len(result.columns) == expected_n_cols


def test_shap_values_raw_no_covariates(no_covariate_data, tmp_path):
    """shap_values() raw=True with no covariates returns only _x and _y SHAP columns"""
    matrix_path, sample_data_path = no_covariate_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
    )
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    result = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    n_snps = len(model._kept_snp_indices_)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "sampleID" in result.columns
    assert (
        len(result.columns) == 1 + n_snps * 2
    )  # sampleID + _x and _y per SNP, no env cols
