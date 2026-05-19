# estimating sample locations from genotype matrices
import allel, re, os, matplotlib, sys, zarr, time, subprocess, copy
import numpy as np, pandas as pd, tensorflow as tf
from scipy import spatial, stats
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import json
from tensorflow.keras import backend as K

tf.keras.utils.set_random_seed(812) #unhash this line to set a random seed for reproducibility, but note that reproducibility is not guaranteed across machines or tensorflow versions.

@tf.keras.utils.register_keras_serializable(name="euclid_loss")
def euclid_loss(y_true, y_pred):
    """Euclidean distance between 2-D coordinates (location)."""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

parser = argparse.ArgumentParser()
parser.add_argument("--vcf", help="VCF with SNPs for all samples.")
parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
parser.add_argument(
    "--matrix",
    help="tab-delimited matrix of minor allele counts with first column named 'sampleID'.\
                                     E.g., \
                                     \
                                     sampleID\tsite1\tsite2\t...\n \
                                     msp1\t0\t1\t...\n \
                                     msp2\t2\t0\t...\n ",
)
parser.add_argument(
    "--sample_data",
    help="tab-delimited text file with columns\
                         'sampleID \t x \t y \t cov1 \t cov2 \t cov3'.\
                          SampleIDs must exactly match those in the \
                          VCF.  \
                          Samples without known locations should \
                          be NA.",
)
parser.add_argument(
    "--train_split",
    default=0.9,
    type=float,
    help="0-1, proportion of samples to use for training. \
                          default: 0.9 ",
)
parser.add_argument(
    "--windows",
    default=False,
    action="store_true",
    help="Run windowed analysis over a single chromosome (requires zarr input).",
)
parser.add_argument("--window_start", default=0, help="default: 0")
parser.add_argument("--window_stop", default=None, help="default: max snp position")
parser.add_argument("--window_size", default=5e5, help="default: 500000")
parser.add_argument(
    "--bootstrap",
    default=False,
    action="store_true",
    help="Run bootstrap replicates by retraining on bootstrapped data.",
)
parser.add_argument(
    "--jacknife",
    default=False,
    action="store_true",
    help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic -- use the bootstrap \
                    option or run windowed analyses for final results.",
)
parser.add_argument(
    "--jacknife_prop",
    default=0.05,
    type=float,
    help="proportion of SNPs to remove for jacknife resampling.\
                    default: 0.05",
)
parser.add_argument(
    "--nboots",
    default=50,
    type=int,
    help="number of bootstrap replicates to run.\
                    default: 50",
)
parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
parser.add_argument("--max_epochs", default=5000, type=int, help="default: 5000")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last \
                          improvement in validation loss. \
                          default: 100",
)
parser.add_argument(
    "--min_mac",
    default=2,
    type=int,
    help="minimum minor allele count.\
                          default: 2.",
)
parser.add_argument(
    "--max_SNPs",
    default=None,
    type=int,
    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.",
)
parser.add_argument(
    "--impute_missing",
    default=False,
    action="store_true",
    help="default: True (if False, all alleles at missing sites are ancestral)",
)
parser.add_argument(
    "--dropout_prop",
    default=0.25,
    type=float,
    help="proportion of weights to zero at the dropout layer. \
                           default: 0.25",
)
parser.add_argument(
    "--nlayers",
    default=10,
    type=int,
    help="number of layers in the network. \
                        default: 10",
)
parser.add_argument(
    "--width",
    default=256,
    type=int,
    help="number of units per layer in the network\
                    default:256",
)
parser.add_argument("--out", help="file name stem for output")
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed for train/test splits and SNP subsetting.",
)
parser.add_argument("--gpu_number", default=None, type=str)
parser.add_argument(
    "--plot_history",
    default=True,
    type=bool,
    help="plot training history? \
                    default: True",
)
parser.add_argument(
    "--gnuplot",
    default=False,
    action="store_true",
    help="print acii plot of training history to stdout? \
                    default: False",
)
parser.add_argument(
    "--keep_weights",
    default=False,
    action="store_true",
    help="keep model weights after training? \
                    default: False.",
)
parser.add_argument(
    "--keep_model",
    default=False,
    action="store_true",
    help="keep model structure and weights after training? \
                    default: False.",
)
parser.add_argument(
    "--load_train_model", 
    default=None,
    type=str,
    help="path to .keras file to load trained model from a previous run."
)
parser.add_argument(
    "--load_params",
    default=None,
    type=str,
    help="Path to a _params.json file to load parameters from a previous run.\
                          Parameters from the json file will supersede all parameters provided \
                          via command line.",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
)
parser.add_argument(
    "--save_metrics",
    default=False,
    action="store_true",
    help="save metrics (R2 & val. error) to file? \
                    default: False",
)
parser.add_argument(
    "--loc_weight",
    default=1.0,
    type=float,
    help="loss weight for euclidean loss function. 1 = default. 0 = no prediction.",
)
parser.add_argument(
    "--env_weight",
    default=1.0,
    type=float,
    help="loss weight for mean squared error loss function. 1 = default. 0 = no prediction",
)
parser.add_argument(
    "--loss",
    default=0.5,
    type=float,
    help="0.5 = default (equal attention to both tasks) 0 = full attention env. cov. prediction, 1 = full attention loc. prediction",
)
parser.add_argument(
    "--num_covs",
    default=3,
    type=int,
    help="number of environmental covariates. default: 3",
)
parser.add_argument(
    "--cov_transforms",
    default=None,
    type=str,
    help="comma-separated transforms for each covariate (e.g. 'none,log,log'). \
          Valid options: none, log. Default: none for all covariates."
)
args = parser.parse_args()
cov_transforms = [t.strip() for t in args.cov_transforms.split(',')] \
    if args.cov_transforms else None

# set seed and gpu
if args.seed is not None:
    np.random.seed(args.seed)
if args.gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

# load old run parameters
if args.load_params is not None:
    with open(args.load_params, "r") as f:
        args.__dict__ = json.load(f)
    f.close()

# store run params
with open(args.out + "_params.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        positions = callset["variants/POS"]
    elif args.vcf is not None:
        print("reading VCF")
        vcf = allel.read_vcf(args.vcf, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif args.matrix is not None:
        gmat = pd.read_csv(args.matrix, sep="\t")
        full_snp_ids = gmat.columns.drop("sampleID").tolist()
        np.savetxt(args.out + sample_id + "_all_snp_ids.txt", full_snp_ids, fmt="%s")
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
    return genotypes, samples

def sort_samples(samples):
    sample_data = pd.read_csv(args.sample_data, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    sample_data = sample_data.reindex(
        np.array(samples)
    )  # sort loc table so samples are in same order as vcf samples
    if not all(
        [sample_data["sampleID2"].iloc[x] == samples[x] for x in range(len(samples))]
    ):  # check that all sample names are present
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()
    locs = np.array(sample_data[["x", "y"] + [f"cov{i+1}" for i in range(args.num_covs)]])
    print(locs)
    print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
    return sample_data, locs

# replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac

def filter_snps(genotypes, samples):
    print("Filtering SNPs...")

    # Start with all SNP indices
    kept_snp_indices_full = np.arange(genotypes.shape[0])  # Original site indices

    # Filter biallelic SNPs
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]
    kept_snp_indices = kept_snp_indices_full[biallel]  # Update SNP indices

    # Filter by minimum minor allele count (MAC)
    if args.min_mac != 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = np.array([x >= args.min_mac for x in derived_counts])
        genotypes = genotypes[ac_filter, :, :]
        kept_snp_indices = kept_snp_indices[ac_filter]  # Update SNP indices

    # Handle missing data imputation
    if args.impute_missing:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]

    # Randomly select a maximum number of SNPs (if specified)
    if args.max_SNPs is not None:
        selected_indices = np.random.choice(len(ac), args.max_SNPs, replace=False)
        ac = ac[selected_indices, :]  # Filter SNPs
        kept_snp_indices = kept_snp_indices[selected_indices]  # Track SNPs

    print(f"Running on {len(ac)} genotypes after filtering.\n")

    # Save SNP indices
    #np.save("test/kept_snp_indices.npy", kept_snp_indices)  
    full_ids = np.loadtxt(args.out + sample_id + "_all_snp_ids.txt", dtype=str)
    kept_ids = full_ids[kept_snp_indices]
    np.savetxt(args.out + sample_id + "_kept_snp_ids.txt", kept_ids, fmt="%s")

    return ac

def normalize_locs(locs, transforms=None, cov_names=None):
    num_covs = locs.shape[1] - 2
    if transforms is None:
        transforms = ['none'] * num_covs
    if cov_names is None:
        cov_names = [f"cov{i+1}" for i in range(num_covs)]
    valid = {'none', 'log'}
    unknown = [t for t in transforms if t not in valid]
    if unknown:
        raise ValueError(f"Unknown transform(s): {unknown}. Valid options: {sorted(valid)}.")
    for i, t in enumerate(transforms):
        if t == 'log':
            col = locs[:, 2 + i]
            nonnan = col[~np.isnan(col)]
            if np.any(nonnan <= 0):
                raise ValueError(
                    f"{cov_names[i]} contains non-positive values "
                    f"(min={nonnan.min():.4f}); log transform requires all values > 0."
                )
    meanlong = np.nanmean(locs[:, 0])
    sdlong   = np.nanstd(locs[:, 0])
    meanlat  = np.nanmean(locs[:, 1])
    sdlat    = np.nanstd(locs[:, 1])
    cov_data = locs[:, 2:].copy().astype(float)
    for i, t in enumerate(transforms):
        if t == 'log':
            cov_data[:, i] = np.log(cov_data[:, i])
    means = np.nanmean(cov_data, axis=0)
    sds   = np.nanstd(cov_data, axis=0)
    for i, t in enumerate(transforms):
        if t == 'none':
            raw_col = locs[:, 2 + i]
            nonnan  = raw_col[~np.isnan(raw_col)]
            if np.all(nonnan > 0):
                gap = nonnan.min() / sds[i]
                skewness = stats.skew(nonnan)
                if gap < 1.0 and skewness > 1.0:
                    print(f"WARNING: {cov_names[i]} is strictly positive but the zero boundary "
                          f"in z-space is only {gap:.2f} units below the training minimum. "
                          f"Predictions may be negative. "
                          f"Consider using 'log' for this covariate.")
    x_norm    = (locs[:, 0] - meanlong) / sdlong
    y_norm    = (locs[:, 1] - meanlat)  / sdlat
    cov_norm  = (cov_data - means) / sds
    norm_locs = np.column_stack([x_norm, y_norm, cov_norm])
    return meanlong, sdlong, meanlat, sdlat, means, sds, transforms, norm_locs

def back_transform_env(z_array, means, sds, transforms):
    result = z_array * sds + means
    for i, t in enumerate(transforms):
        if t == 'log':
            result[:, i] = np.exp(result[:, i])
    return result

def split_train_test(ac, locs, samples):
    train = np.argwhere(~np.isnan(locs[:, 0])).flatten()
    pred = np.array([x for x in range(len(locs)) if x not in train])
    if pred.size == 0:
        print("No samples found for prediction! At least one sample for prediction needed.")
        sys.exit(1)
    test = np.random.choice(train, round((1 - args.train_split) * len(train)), replace=False)
    train = np.array([x for x in train if x not in test])

    # Save sample IDs that were selected
    train_sample_ids = samples[train]
    test_sample_ids = samples[test]
    pred_sample_ids = samples[pred]

    np.savetxt(args.out + sample_id + "_kept_train_samples.txt", train_sample_ids, fmt='%s')  # Save training sample IDs
    np.savetxt(args.out + sample_id + "_kept_test_samples.txt", test_sample_ids, fmt='%s')    # Save testing sample IDs
    np.savetxt(args.out + sample_id + "_kept_pred_samples.txt", pred_sample_ids, fmt='%s')  # Save prediction sample IDs   

    traingen = np.transpose(ac[:, train])
    trainlocs = [locs[train][:, 0:2], locs[train][:, 2:]]
    testgen = np.transpose(ac[:, test])
    testlocs = [locs[test][:, 0:2], locs[test][:, 2:]]
    predgen = np.transpose(ac[:, pred])

    np.save(args.out + sample_id + "_traingen_array.npy", traingen)
    np.save(args.out + sample_id + "_testgen_array.npy",  testgen)
    np.save(args.out + sample_id + "_predgen_array.npy",  predgen) 

    return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen

def load_network_dual(traingen):
    """
    creates and loads a neural network with two outputs
    one for the location and one for the environmental covariates
    """
    geno_input = tf.keras.Input(shape=(traingen.shape[1],), name="geno_input")
    trunk_model = tf.keras.layers.BatchNormalization()(geno_input)
    for i in range(int(np.floor(args.nlayers / 2))):
        trunk_model = tf.keras.layers.Dense(args.width, activation="elu")(trunk_model)
    trunk_model = tf.keras.layers.Dropout(args.dropout_prop)(trunk_model)
    for i in range(int(np.ceil(args.nlayers / 2))):
        trunk_model = tf.keras.layers.Dense(args.width, activation="elu")(trunk_model)
    loc_model = tf.keras.layers.Dense(2)(trunk_model)
    loc_output = tf.keras.layers.Dense(2)(loc_model)
    env_model = tf.keras.layers.Dense(args.num_covs)(trunk_model)
    env_output = tf.keras.layers.Dense(args.num_covs)(env_model)
    model = tf.keras.Model(inputs=geno_input, outputs=[loc_output, env_output])
    model.compile(optimizer="Adam", loss=[euclid_loss, "mse"], loss_weights=[args.loc_weight, args.env_weight])
    model.summary()
    return model

def get_sample_id(input_file):
    base_name = os.path.basename(input_file)  # Get the base name, e.g., 'masked_data_RC01_A10_5144.tsv'
    sample_id = base_name.replace('masked_data_', '').replace('.tsv', '')  # Extract the sample ID, e.g., 'RC01_A10_5144'
    return sample_id

sample_id = get_sample_id(args.sample_data)

def load_callbacks(boot):
    if args.keep_model:
        if args.bootstrap or args.jacknife:
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + "_boot" + str(boot) + "_model.keras",
            verbose=args.keras_verbose,
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            save_freq='epoch', 
        )
        else: 
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + sample_id + "_model.keras",
            verbose=args.keras_verbose,
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            save_freq='epoch',
        )
    else:
        if args.bootstrap or args.jacknife:
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + "_boot" + str(boot) + ".weights.h5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq='epoch',
        )
        else:
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + sample_id + ".weights.h5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq='epoch',
        )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience / 6),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return checkpointer, earlystop, reducelr

def train_network(model, traingen, testgen, trainlocs, testlocs):
    print("[DEBUG] train_network called")
    if args.load_train_model and os.path.isfile(args.load_train_model):
        print("[DEBUG] SKIPPING model.fit(); using saved weights")
        model = tf.keras.models.load_model(
            args.load_train_model,
            custom_objects={"euclid_loss": euclid_loss},
            compile=False,
        )
        return tf.keras.callbacks.History(), model
    
    history = model.fit(
        traingen,
        trainlocs,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=args.keras_verbose,
        validation_data=(testgen, testlocs),
        callbacks=[checkpointer, earlystop, reducelr],
    )
    if args.keep_model:
        if args.bootstrap or args.jacknife:
            tf.keras.models.load_model(args.out + "_boot" + str(boot) + "_model.keras")
        else: 
            tf.keras.models.load_model(args.out + sample_id + "_model.keras")
    elif args.load_train_model is not None:
        model = tf.keras.models.load_model(args.load_train_model)
    else:
        if args.bootstrap or args.jacknife:
            model.load_weights(args.out + "_boot" + str(boot) + ".weights.h5")
        else:
            model.load_weights(args.out + sample_id + ".weights.h5")
    return history, model

def predict_locs(
    model,
    predgen,
    sdlong,
    meanlong,
    sdlat,
    meanlat,
    sds,
    means,
    testlocs,
    pred,
    samples,
    testgen,
    transforms,
    verbose=True,
):
    if verbose == True:
        print("predicting locations...")
    prediction = model.predict(predgen)
    prediction_longlat = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in prediction[0]]          
    )
    prediction_env = back_transform_env(prediction[1], means, sds, transforms)
    predi = pd.DataFrame(prediction_longlat, columns= ['x', 'y'])
    ction = pd.DataFrame(prediction_env, columns = [f"cov{i+1}" for i in range(args.num_covs)])
    prediction = [predi, ction]
    predout = pd.concat((prediction), axis=1)
    predout["sampleID"] = samples[pred]
    testlocs_flat = np.concatenate([testlocs[0], testlocs[1]], axis=1)
    if args.bootstrap or args.jacknife:
        predout.to_csv(args.out + "_boot" + str(boot) + "_predlocs.txt", index=False)
        testlocs2 = np.column_stack([
            testlocs_flat[:, 0] * sdlong + meanlong,
            testlocs_flat[:, 1] * sdlat  + meanlat,
            back_transform_env(testlocs_flat[:, 2:], means, sds, transforms),
        ])
    elif args.windows:
        predout.to_csv(
            args.out + "_" + str(i) + "-" + str(i + size - 1) + "_predlocs.txt",
            index=False,
        )  
        testlocs2 = np.column_stack([
            testlocs_flat[:, 0] * sdlong + meanlong,
            testlocs_flat[:, 1] * sdlat  + meanlat,
            back_transform_env(testlocs_flat[:, 2:], means, sds, transforms),
        ])
    else:
        predout.to_csv(args.out + sample_id + "_predlocs.txt", index=False)
        testlocs2 = np.column_stack([
            testlocs_flat[:, 0] * sdlong + meanlong,
            testlocs_flat[:, 1] * sdlat  + meanlat,
            back_transform_env(testlocs_flat[:, 2:], means, sds, transforms),
        ])
    p2 = model.predict(testgen)  # print validation loss to screen
    p2 = np.concatenate([p2[0], p2[1]], axis=1)
    p2 = np.column_stack([
        p2[:, 0] * sdlong + meanlong,
        p2[:, 1] * sdlat  + meanlat,
        back_transform_env(p2[:, 2:], means, sds, transforms),
    ])
    r2_long = np.corrcoef(p2[:, 0], testlocs2[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(p2[:, 1], testlocs2[:, 1])[0][1] ** 2
    r2_covs = [np.corrcoef(p2[:, 2 + i], testlocs2[:, 2 + i])[0][1] ** 2 for i in range(args.num_covs)]
    mean_dist = np.mean(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    median_dist = np.median(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    dists = [
        spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))
    ]
    if args.save_metrics:
        results_data = {
            "Sample_ID": sample_id,
            "R2_x": r2_long,
            "R2_y": r2_lat,
            **{f"R2_cov{i+1}": r2_covs[i] for i in range(args.num_covs)},
            "Mean_Validation_Error": mean_dist,
            "Median_Validation_Error": median_dist
        }
        results_df = pd.DataFrame(results_data, index=[0])
        metrics_file = os.path.join(args.out + "_metrics.txt")
        if os.path.exists(metrics_file):
            results_df.to_csv(metrics_file, sep="\t", index=False, mode="a", header=False)
        else:
            results_df.to_csv(metrics_file, sep="\t", index=False, mode='w')
    if verbose == True:
        print(
            f"R2(x)={r2_long}\nR2(y)={r2_lat}\n" + "\n".join([f"R2(cov{i+1})={r2_covs[i]}" for i in range(args.num_covs)]) + "\n" +
            "mean validation error " + str(mean_dist) + "\n" +
            "median validation error " + str(median_dist) + "\n"
        )
    hist = pd.DataFrame(history.history)
    hist.to_csv(args.out + sample_id + "_history.txt", sep="\t", index=False)
    return dists

def plot_history(history, dists, gnuplot):
    if args.plot_history:
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 1.5), dpi=200)
        plt.rcParams.update({"font.size": 7})
        ax1 = fig.add_axes([0, 0, 0.4, 1])
        ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
        ax1.set_xlabel("Validation Loss")
        ax2 = fig.add_axes([0.55, 0, 0.4, 1])
        ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
        ax2.set_xlabel("Training Loss")
        fig.savefig(args.out + sample_id + "_fitplot.pdf", bbox_inches="tight")
        if gnuplot:
            gp.plot(
                np.array(history.history["val_loss"][3:]),
                unset="grid",
                terminal="dumb 60 20",
                # set= 'logscale y',
                title="Validation Loss by Epoch",
            )
            gp.plot(
                (np.array(dists), dict(histogram="freq", binwidth=np.std(dists) / 5)),
                unset="grid",
                terminal="dumb 60 20",
                title="Test Error",
            )

# Code that runs the analysis
if args.windows:
    callset = zarr.open_group(args.zarr, mode="r")
    gt = callset["calldata/GT"]
    samples = callset["samples"][:]
    positions = np.array(callset["variants/POS"])
    start = int(args.window_start)
    if args.window_stop == None:
        stop = np.max(positions)
    else:
        stop = int(args.window_stop)
    size = int(args.window_size)
    for i in np.arange(start, stop, size):
        mask = np.logical_and(positions >= i, positions < i + size)
        a = np.min(np.argwhere(mask))
        b = np.max(np.argwhere(mask))
        print(a, b)
        genotypes = allel.GenotypeArray(gt[a:b, :, :])
        sample_data, locs = sort_samples(samples)
        cov_names = [f"cov{i+1}" for i in range(args.num_covs)]
        meanlong, sdlong, meanlat, sdlat, means, sds, transforms, locs = normalize_locs(
            locs, transforms=cov_transforms, cov_names=cov_names
        )
        ac = filter_snps(genotypes, samples)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network_dual(traingen)
        t1 = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            sds,
            means,
            testlocs,
            pred,
            samples,
            testgen,
            transforms,
        )
        plot_history(history, dists, args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm " + args.out + ".weights.h5", shell=True)
        t2 = time.time()
        elapsed = t2 - t1
        print("run time " + str(elapsed / 60) + " minutes")
else:
    if not args.bootstrap and not args.jacknife:
        boot = None
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        cov_names = [f"cov{i+1}" for i in range(args.num_covs)]
        meanlong, sdlong, meanlat, sdlat, means, sds, transforms, locs = normalize_locs(
            locs, transforms=cov_transforms, cov_names=cov_names
        )
        ac = filter_snps(genotypes, samples)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs, samples)
        model = load_network_dual(traingen)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        #np.save("test/traingen_array.npy", traingen)
        #np.save("test/testgen_array.npy", testgen)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat, 
            sds,
            means,
            testlocs,
            pred,
            samples,
            testgen,
            transforms,
        )
        plot_history(history, dists, args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm " + args.out + sample_id + ".weights.h5", shell=True)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
    elif args.bootstrap:
        boot = "FULL"
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        cov_names = [f"cov{i+1}" for i in range(args.num_covs)]
        meanlong, sdlong, meanlat, sdlat, means, sds, transforms, locs = normalize_locs(
            locs, transforms=cov_transforms, cov_names=cov_names
        )
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network_dual(traingen)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            sds,
            means,
            testlocs,
            pred,
            samples,
            testgen,
            transforms,
        )
        plot_history(history, dists, args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm " + args.out + "_bootFULL.weights.h5", shell=True)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        for boot in range(args.nboots):
            np.random.seed(np.random.choice(range(int(1e6)), 1))
            checkpointer, earlystop, reducelr = load_callbacks(boot)
            print("starting bootstrap " + str(boot))
            traingen2 = copy.deepcopy(traingen)
            testgen2 = copy.deepcopy(testgen)
            predgen2 = copy.deepcopy(predgen)
            site_order = np.random.choice(
                traingen2.shape[1], traingen2.shape[1], replace=True
            )
            traingen2 = traingen2[:, site_order]
            testgen2 = testgen2[:, site_order]
            predgen2 = predgen2[:, site_order]
            model = load_network_dual(traingen2)
            start = time.time()
            history, model = train_network(
                model, traingen2, testgen2, trainlocs, testlocs
            )
            dists = predict_locs(
                model,
                predgen2,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                sds,
                means,
                testlocs,
                pred,
                samples,
                testgen2,
            )
            plot_history(history, dists, args.gnuplot)
            if not args.keep_weights:
                subprocess.run(
                    "rm " + args.out + "_boot" + str(boot) + ".weights.h5", shell=True
                )
            end = time.time()
            elapsed = end - start
            K.clear_session()
            print("run time " + str(elapsed / 60) + " minutes\n\n")
    elif args.jacknife:
        boot = "FULL"
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        cov_names = [f"cov{i+1}" for i in range(args.num_covs)]
        meanlong, sdlong, meanlat, sdlat, means, sds, transforms, locs = normalize_locs(
            locs, transforms=cov_transforms, cov_names=cov_names
        )
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks(boot)
        (
            train,
            test,
            traingen,
            testgen,
            trainlocs,
            testlocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network_dual(traingen)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            sds,
            means,
            testlocs,
            pred,
            samples,
            testgen,
            transforms,
        )
        plot_history(history, dists, args.gnuplot)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        print("starting jacknife resampling")
        af = []
        for i in tqdm(range(ac.shape[0])):
            af.append(sum(ac[i, :]) / (ac.shape[1] * 2))
        af = np.array(af)
        for boot in tqdm(range(args.nboots)):
            checkpointer, earlystop, reducelr = load_callbacks(boot)
            pg = copy.deepcopy(predgen)
            sites_to_remove = np.random.choice(
                pg.shape[1], int(pg.shape[1] * args.jacknife_prop), replace=False
            )
            for i in sites_to_remove:
                pg[:, i] = np.random.binomial(2, af[i], pg.shape[0])
            dists = predict_locs(
                model,
                pg,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                sds,
                means,
                testlocs,
                pred,
                samples,
                testgen,
                verbose=False,
            )
        if not args.keep_weights:
            subprocess.run("rm " + args.out + "_bootFULL.weights.h5", shell=True)
