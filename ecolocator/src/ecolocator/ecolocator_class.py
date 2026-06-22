import numpy as np
import pandas as pd
import os
import json
import logging
import tensorflow as tf
from .utils import (
    load_genotypes,
    sort_samples,
    normalize_locs,
    filter_snps,
    replace_missing_data,
    back_transform_env,
)
from .network import build_network, train_network, euclid_loss


class EcoLocator:
    """
    A neural network model for predicting sample locations and environmental
    covariates from genotype data.
    """

    def __init__(
        self,
        cov_transforms: list = None,
        nlayers: int = 10,
        width: int = 256,
        dropout_prop: float = 0.25,
        loc_weight: float = 1.0,
        env_weight: float = 1.0,
    ):
        self.cov_transforms = cov_transforms
        self.nlayers = nlayers
        self.width = width
        self.dropout_prop = dropout_prop
        self.loc_weight = loc_weight
        self.env_weight = env_weight

    @classmethod
    def load(cls, path: str) -> "EcoLocator":
        with open(os.path.join(path, "params.json")) as f:
            params = json.load(f)

        obj = cls(
            cov_transforms=params["cov_transforms"],
            nlayers=params["nlayers"],
            width=params["width"],
            dropout_prop=params["dropout_prop"],
            loc_weight=params["loc_weight"],
            env_weight=params["env_weight"],
        )

        arrays = np.load(os.path.join(path, "arrays.npz"))
        obj.meanlong_ = arrays["meanlong"].item()
        obj.sdlong_ = arrays["sdlong"].item()
        obj.meanlat_ = arrays["meanlat"].item()
        obj.sdlat_ = arrays["sdlat"].item()
        obj.means_ = arrays["means"]
        obj.sds_ = arrays["sds"]
        obj._kept_snp_indices_ = arrays["kept_snp_indices"]

        obj.cov_names_ = params["cov_names"]
        obj.seed_ = params["seed"]
        obj.num_covs_ = len(obj.cov_names_)
        obj.transforms_ = params["transforms"]

        obj.model_ = tf.keras.models.load_model(
            os.path.join(path, "model.keras"),
            custom_objects={"euclid_loss": euclid_loss},
        )

        return obj

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model_.save(os.path.join(path, "model.keras"))
        np.savez(
            os.path.join(path, "arrays.npz"),
            meanlong=self.meanlong_,
            sdlong=self.sdlong_,
            meanlat=self.meanlat_,
            sdlat=self.sdlat_,
            means=self.means_,
            sds=self.sds_,
            kept_snp_indices=self._kept_snp_indices_,
        )
        params = {
            "seed": self.seed_,
            "cov_names": self.cov_names_,
            "num_covs": self.num_covs_,
            "transforms": self.transforms_,
            "cov_transforms": self.cov_transforms,
            "nlayers": self.nlayers,
            "width": self.width,
            "dropout_prop": self.dropout_prop,
            "loc_weight": self.loc_weight,
            "env_weight": self.env_weight,
        }
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(params, f)

    def fit(
        self,
        genotype_path: str,
        sample_data_path: str,
        max_epochs: int = 5000,
        patience: int = 100,
        batch_size: int = 32,
        min_mac: int = 2,
        max_snps: int = None,
        train_split: float = 0.9,
        seed: int = None,
        verbose: int = 1,
    ) -> "EcoLocator":
        rng = np.random.default_rng(seed)

        genotypes, samples = self._get_genotypes(genotype_path)
        sample_data, locs = sort_samples(samples, sample_data_path)

        num_covs = locs.shape[1] - 2
        cov_names = [c for c in sample_data.columns if c not in {"sampleID2", "x", "y"}]
        self.cov_names_ = cov_names
        if num_covs < 1:
            logging.warning(
                f"Found {num_covs} covariates in sample data. Proceeding with location data only."
            )

        (
            self.meanlong_,
            self.sdlong_,
            self.meanlat_,
            self.sdlat_,
            self.means_,
            self.sds_,
            self.transforms_,
            locs,
        ) = normalize_locs(locs, transforms=self.cov_transforms, cov_names=cov_names)

        genotypes, self._kept_snp_indices_ = filter_snps(
            genotypes, min_mac=min_mac, max_snps=max_snps, rng=rng
        )
        ac = replace_missing_data(genotypes, rng=rng)

        train = np.argwhere(~np.isnan(locs[:, 0])).flatten()
        test = rng.choice(train, round((1 - train_split) * len(train)), replace=False)
        train = np.array([x for x in train if x not in test])

<<<<<<< HEAD
        traingen = np.transpose(ac[:, train])
        testgen = np.transpose(ac[:, test])
        trainlocs = [locs[train][:, 0:2], locs[train][:, 2:]]
        testlocs = [locs[test][:, 0:2], locs[test][:, 2:]]
=======
        traingen  = np.transpose(ac[:, train])
        testgen   = np.transpose(ac[:, test])

        if num_covs > 0:
            trainlocs = [locs[train][:, 0:2], locs[train][:, 2:]]
            testlocs  = [locs[test][:, 0:2],  locs[test][:, 2:]]
        else:
            trainlocs = locs[train][:, 0:2]
            testlocs  = locs[test][:, 0:2]
>>>>>>> 7bc048d (updates too allow location only inputs)

        self.num_covs_ = num_covs
        self.seed_ = seed
        self.model_ = build_network(
            n_snps=traingen.shape[1],
            num_covs=num_covs,
            nlayers=self.nlayers,
            width=self.width,
            dropout_prop=self.dropout_prop,
            loc_weight=self.loc_weight,
            env_weight=self.env_weight,
        )
        self.history_ = train_network(
            self.model_,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=verbose,
        )
        return self

    def predict(
        self,
        genotype_path: str,
        sample_data_path: str,
    ) -> pd.DataFrame:
        genotypes, samples = self._get_genotypes(genotype_path)
        _, locs = sort_samples(samples, sample_data_path)

        genotypes = genotypes[self._kept_snp_indices_, :, :]
        ac = replace_missing_data(genotypes)

        pred = np.argwhere(np.isnan(locs[:, 0])).flatten()
        predgen = np.transpose(ac[:, pred])

        prediction = self.model_.predict(predgen)
        if self.num_covs_ > 0:
            pred_longlat = (
                prediction[0] * np.array([self.sdlong_, self.sdlat_])
                + np.array([self.meanlong_, self.meanlat_])
            )
            pred_env = back_transform_env(
                prediction[1], self.means_, self.sds_, self.transforms_
            )

<<<<<<< HEAD
        pred_longlat = prediction[0] * np.array([self.sdlong_, self.sdlat_]) + np.array(
            [self.meanlong_, self.meanlat_]
        )
        pred_env = back_transform_env(
            prediction[1], self.means_, self.sds_, self.transforms_
        )

        result = pd.DataFrame(pred_longlat, columns=["x", "y"])
        cov_cols = pd.DataFrame(pred_env, columns=self.cov_names_)
        result = pd.concat([result, cov_cols], axis=1)
        result.insert(0, "sampleID", samples[pred])
        return result
=======
            result = pd.DataFrame(pred_longlat, columns=["x", "y"])
            cov_cols = pd.DataFrame(pred_env, columns=self.cov_names_)
            result = pd.concat([result, cov_cols], axis=1)
            result.insert(0, "sampleID", samples[pred])
            return result
        else:
            pred_longlat = (
                prediction * np.array([self.sdlong_, self.sdlat_])
                + np.array([self.meanlong_, self.meanlat_])
            )
            
            result = pd.DataFrame(pred_longlat, columns=["x", "y"])
            result.insert(0, "sampleID", samples[pred])
            return result
>>>>>>> 7bc048d (updates too allow location only inputs)

    def fit_predict_loo(
        self,
        genotype_path: str,
        sample_data_path: str,
        max_epochs: int = 5000,
        patience: int = 100,
        batch_size: int = 32,
        min_mac: int = 2,
        max_snps: int = None,
        max_folds: int = None,
        train_split: float = 0.9,
        seed: int = None,
        verbose: int = 1,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        genotypes, samples = self._get_genotypes(genotype_path)
        sample_data, locs = sort_samples(samples, sample_data_path)

        num_covs = locs.shape[1] - 2
        cov_names = [c for c in sample_data.columns if c not in {"sampleID2", "x", "y"}]

        genotypes, _ = filter_snps(
            genotypes, min_mac=min_mac, max_snps=max_snps, rng=rng
        )
        ac = replace_missing_data(genotypes, rng=rng)

        known = np.argwhere(~np.isnan(locs[:, 0])).flatten()
        if max_folds is not None:
            known = known[:max_folds]
        all_predictions = []

        for fold, i in enumerate(known, start=1):
            logging.info(
                f"LOO iteration {fold} of {len(known)}: holding out {samples[i]}"
            )
            loo_locs = locs.copy()
            loo_locs[i] = np.nan

<<<<<<< HEAD
            (meanlong, sdlong, meanlat, sdlat, means, sds, transforms, norm_locs) = (
                normalize_locs(
                    loo_locs, transforms=self.cov_transforms, cov_names=cov_names
                )
=======
            
            (meanlong, sdlong, meanlat, sdlat,
            means, sds, transforms, norm_locs) = normalize_locs(
                loo_locs, transforms=self.cov_transforms, cov_names=cov_names
>>>>>>> 7bc048d (updates too allow location only inputs)
            )

            train = np.argwhere(~np.isnan(norm_locs[:, 0])).flatten()
            test = rng.choice(
                train, round((1 - train_split) * len(train)), replace=False
            )
            train = np.array([x for x in train if x not in test])

<<<<<<< HEAD
            traingen = np.transpose(ac[:, train])
            testgen = np.transpose(ac[:, test])
            trainlocs = [norm_locs[train][:, 0:2], norm_locs[train][:, 2:]]
            testlocs = [norm_locs[test][:, 0:2], norm_locs[test][:, 2:]]
            predgen = np.transpose(ac[:, [i]])
=======
            traingen  = np.transpose(ac[:, train])
            testgen   = np.transpose(ac[:, test])

            if num_covs > 0:
                trainlocs = [norm_locs[train][:, 0:2], norm_locs[train][:, 2:]]
                testlocs  = [norm_locs[test][:, 0:2],  norm_locs[test][:, 2:]]
            else:
                trainlocs = norm_locs[train][:, 0:2]
                testlocs  = norm_locs[test][:, 0:2]

            predgen   = np.transpose(ac[:, [i]])
>>>>>>> 7bc048d (updates too allow location only inputs)

            model = build_network(
                n_snps=traingen.shape[1],
                num_covs=num_covs,
                nlayers=self.nlayers,
                width=self.width,
                dropout_prop=self.dropout_prop,
                loc_weight=self.loc_weight,
                env_weight=self.env_weight,
            )
            train_network(
                model,
                traingen,
                testgen,
                trainlocs,
                testlocs,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )

<<<<<<< HEAD
            prediction = model.predict(predgen)
            pred_longlat = prediction[0] * np.array([sdlong, sdlat]) + np.array(
                [meanlong, meanlat]
            )
            pred_env = back_transform_env(prediction[1], means, sds, transforms)

            row = {
                "sampleID": samples[i],
                "x": pred_longlat[0, 0],
                "y": pred_longlat[0, 1],
            }
            for j, name in enumerate(cov_names):
                row[name] = pred_env[0, j]
            all_predictions.append(row)
            del model
=======
            prediction  = model.predict(predgen)
            if num_covs > 0:
                pred_longlat = (
                    prediction[0] * np.array([sdlong, sdlat])
                    + np.array([meanlong, meanlat])
                )
                pred_env = back_transform_env(prediction[1], means, sds, transforms)

                row = {"sampleID": samples[i], "x": pred_longlat[0, 0], "y": pred_longlat[0, 1]}
                for j, name in enumerate(cov_names):
                    row[name] = pred_env[0, j]
                all_predictions.append(row)
            else:
                pred_longlat = (
                    prediction * np.array([sdlong, sdlat])
                    + np.array([meanlong, meanlat])
                )

                row = {"sampleID": samples[i], "x": pred_longlat[0, 0], "y": pred_longlat[0, 1]}
                all_predictions.append(row)

            del model 
>>>>>>> 7bc048d (updates too allow location only inputs)
            tf.keras.backend.clear_session()

        return pd.DataFrame(all_predictions)

    def shap_values(
        self,
        genotype_path: str,
        sample_data_path: str,
        train_genotype_path: str,
        train_sample_data_path: str,
        background_size: int = 100,
        min_maf: float = None,
        seed: int = None,
        raw: bool = False,
    ) -> pd.DataFrame:
        import shap

        if not hasattr(self, "model_"):
            raise RuntimeError("EcoLocator must be fitted before calling shap_values")

        rng = np.random.default_rng(seed if seed is not None else self.seed_)

        # load and filter training genos for bg
        train_genotypes, train_samples = self._get_genotypes(train_genotype_path)
        train_genotypes = train_genotypes[self._kept_snp_indices_, :, :]
        train_ac = replace_missing_data(train_genotypes, rng=rng)
        _, train_locs = sort_samples(train_samples, train_sample_data_path)
        known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
        traingen = np.transpose(train_ac[:, known])

        # load and filter pred genos
        pred_genotypes, pred_samples = self._get_genotypes(genotype_path)
        pred_genotypes = pred_genotypes[self._kept_snp_indices_, :, :]
        pred_ac = replace_missing_data(pred_genotypes)
        _, pred_locs = sort_samples(pred_samples, sample_data_path)
        unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
        predgen = np.transpose(pred_ac[:, unknown])

        # build bg
        bg_size = min(background_size, traingen.shape[0])
        bg_idx = rng.choice(traingen.shape[0], bg_size, replace=False)
        background = traingen[bg_idx, :]

<<<<<<< HEAD
        # wrap model heads
        model_loc = tf.keras.Model(
            inputs=self.model_.input, outputs=self.model_.output[0]
        )
        model_env = tf.keras.Model(
            inputs=self.model_.input, outputs=self.model_.output[1]
        )
=======
        #wrap model heads
        if self.num_covs_ > 0:
            model_loc = tf.keras.Model(inputs=self.model_.input, outputs=self.model_.output[0])
        else:
            model_loc = self.model_
>>>>>>> 7bc048d (updates too allow location only inputs)

        # compute shap
        expl_loc = shap.GradientExplainer(model_loc, background)
        shap_loc = expl_loc.shap_values(predgen)

        if self.num_covs_ > 0:
            model_env = tf.keras.Model(inputs=self.model_.input, outputs=self.model_.output[1])
            expl_env = shap.GradientExplainer(model_env, background)
            shap_env = expl_env.shap_values(predgen)
        
        
        snp_ids = self._kept_snp_indices_

        # apply min_maf filter to snp index list before building either format
        if min_maf is not None:
            af = np.mean(traingen, axis=0) / 2.0
            maf = np.minimum(af, 1 - af)
            mask = maf >= min_maf
            snp_ids = snp_ids[mask]
            shap_loc = shap_loc[:, mask, :]
            if self.num_covs_ > 0:
                shap_env = shap_env[:, mask, :]

        n_samples = predgen.shape[0]
        if n_samples == 1:
            logging.warning(
                "Attributing 1 sample — output is absolute SHAP values "
                "for that sample, not a mean across multiple samples."
            )

        if raw:
            rows = []
            for k, sample in enumerate(pred_samples[unknown]):
                row = {"sampleID": sample}
                for col_idx, snp_id in enumerate(snp_ids):
                    row[f"{snp_id}_x"] = shap_loc[k, col_idx, 0]
                    row[f"{snp_id}_y"] = shap_loc[k, col_idx, 1]
                    for j, cov in enumerate(self.cov_names_):
                        row[f"{snp_id}_{cov}"] = shap_env[k, col_idx, j]
                rows.append(row)
            return pd.DataFrame(rows)

        # default: mean absolute SHAP summary — rows=SNPs, cols=output variables
        output_cols = ["x", "y"] + self.cov_names_
        summary = {"snp_id": snp_ids}
        for var_idx, var in enumerate(output_cols):
            if var_idx < 2:
                vals = shap_loc[:, :, var_idx]  # shape (n_samples, n_snps)
            else:
                vals = shap_env[:, :, var_idx - 2]
            summary[var] = np.abs(vals).mean(axis=0)
        return pd.DataFrame(summary)

    def _get_genotypes(self, genotype_path: str):
        if genotype_path.endswith(".zarr"):
            return load_genotypes(zarr_path=genotype_path)
        elif genotype_path.endswith((".vcf", ".vcf.gz")):
            return load_genotypes(vcf_path=genotype_path)
        else:
            return load_genotypes(matrix_path=genotype_path)
