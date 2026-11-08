import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from ecolocator import EcoLocator
from ecolocator.utils import replace_missing_data, sort_samples


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


def test_fit_predict_loo_fold_range(example_data):
    """fold_start/fold_stop hold out exactly the samples in that fold range"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    result = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
        fold_start=2,
        fold_stop=4,
    )
    full_ids = model.fit_predict_loo(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=1,
        patience=1,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )["sampleID"].tolist()

    assert result["sampleID"].tolist() == full_ids[2:4]


def test_fit_predict_loo_chunk_matches_full_run(example_data):
    """With a seed, a fold run alone gives the same prediction as in a full run"""
    _, _, matrix_path, sample_data_path = example_data
    kwargs = dict(max_epochs=5, patience=3, train_split=0.6, min_mac=1, seed=7)
    full = EcoLocator(nlayers=2, width=32).fit_predict_loo(
        str(matrix_path), str(sample_data_path), fold_stop=3, **kwargs
    )
    chunk = EcoLocator(nlayers=2, width=32).fit_predict_loo(
        str(matrix_path), str(sample_data_path), fold_start=2, fold_stop=3, **kwargs
    )

    pd.testing.assert_frame_equal(
        full.iloc[[2]].reset_index(drop=True), chunk.reset_index(drop=True)
    )


def test_fit_seed_is_reproducible(tmp_path, example_data):
    """fit() with the same seed gives the same predictions"""
    _, _, matrix_path, sample_data_path = example_data
    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)
    preds = []
    for _ in range(2):
        model = EcoLocator(nlayers=2, width=32)
        model.fit(
            str(matrix_path),
            str(masked_path),
            max_epochs=5,
            patience=3,
            train_split=0.6,
            min_mac=1,
            seed=3,
            verbose=0,
        )
        preds.append(model.predict(str(matrix_path), str(masked_path)))

    pd.testing.assert_frame_equal(preds[0], preds[1])


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


def test_shap_values_raw_satisfies_additivity_location(example_data, tmp_path):
    """test that signed per-SNP SHAP values for the location head sum to (model output - baseline)"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    raw = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    assert len(raw) == 1

    train_genotypes, train_samples, train_snp_ids = model._get_genotypes(
        str(matrix_path)
    )
    train_genotypes = train_genotypes[model._kept_snp_indices_, :, :]
    train_ac = replace_missing_data(
        train_genotypes, rng=np.random.default_rng(model.seed_)
    )
    _, train_locs = sort_samples(train_samples, str(sample_data_path))
    known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
    traingen = np.transpose(train_ac[:, known]).astype(np.float32)

    pred_genotypes, pred_samples, _ = model._get_genotypes(str(matrix_path))
    pred_genotypes = pred_genotypes[model._kept_snp_indices_, :, :]
    pred_ac = replace_missing_data(pred_genotypes)
    _, pred_locs = sort_samples(pred_samples, str(masked_path))
    unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
    predgen = np.transpose(pred_ac[:, unknown]).astype(np.float32)

    snp_ids = train_snp_ids[model._kept_snp_indices_]

    model_loc = tf.keras.Model(
        inputs=model.model_.input, outputs=model.model_.output[0]
    )
    raw_loc = model_loc.predict(predgen, verbose=0)
    baseline_loc = model_loc.predict(traingen, verbose=0).mean(axis=0)

    row = raw.iloc[0]
    for out_idx, out_name in enumerate(["x", "y"]):
        shap_sum = sum(row[f"{snp_id}_{out_name}"] for snp_id in snp_ids)
        identity = shap_sum + baseline_loc[out_idx]
        assert np.isclose(identity, raw_loc[0, out_idx], atol=2.0)


def test_shap_values_raw_satisfies_additivity_env(example_data, tmp_path):
    """test that signed per-SNP SHAP values for the env head sum to (model output - baseline)"""
    _, _, matrix_path, sample_data_path = example_data
    model = EcoLocator(nlayers=2, width=32)
    model.fit(
        str(matrix_path),
        str(sample_data_path),
        max_epochs=5,
        patience=3,
        train_split=0.6,
        min_mac=1,
        seed=1,
    )

    sample_data = pd.read_csv(sample_data_path, sep="\t")
    sample_data.iloc[0, 1:] = np.nan
    masked_path = tmp_path / "masked.tsv"
    sample_data.to_csv(masked_path, sep="\t", index=False)

    raw = model.shap_values(
        str(matrix_path),
        str(masked_path),
        str(matrix_path),
        str(sample_data_path),
        raw=True,
    )
    assert len(raw) == 1

    train_genotypes, train_samples, train_snp_ids = model._get_genotypes(
        str(matrix_path)
    )
    train_genotypes = train_genotypes[model._kept_snp_indices_, :, :]
    train_ac = replace_missing_data(
        train_genotypes, rng=np.random.default_rng(model.seed_)
    )
    _, train_locs = sort_samples(train_samples, str(sample_data_path))
    known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
    traingen = np.transpose(train_ac[:, known]).astype(np.float32)

    pred_genotypes, pred_samples, _ = model._get_genotypes(str(matrix_path))
    pred_genotypes = pred_genotypes[model._kept_snp_indices_, :, :]
    pred_ac = replace_missing_data(pred_genotypes)
    _, pred_locs = sort_samples(pred_samples, str(masked_path))
    unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
    predgen = np.transpose(pred_ac[:, unknown]).astype(np.float32)

    snp_ids = train_snp_ids[model._kept_snp_indices_]

    model_env = tf.keras.Model(
        inputs=model.model_.input, outputs=model.model_.output[1]
    )
    raw_env = model_env.predict(predgen, verbose=0)
    baseline_env = model_env.predict(traingen, verbose=0).mean(axis=0)

    row = raw.iloc[0]
    for out_idx, cov in enumerate(model.cov_names_):
        shap_sum = sum(row[f"{snp_id}_{cov}"] for snp_id in snp_ids)
        identity = shap_sum + baseline_env[out_idx]
        assert np.isclose(identity, raw_env[0, out_idx], atol=2.0)
