import numpy as np
import pandas as pd
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
    assert list(result.columns) == ["x", "y", "cov1", "cov2", "cov3", "sampleID"]
    assert len(result) == 1

def test_predict_positive_with_log_transform(example_data, tmp_path):
    """predict() with log transform always returns positive covariate values"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32, cov_transforms=['none', 'none', 'log'])
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
    assert list(result.columns) == ["x", "y", "cov1", "cov2", "cov3", "sampleID"]