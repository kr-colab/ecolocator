import numpy as np
import pandas as pd
import pytest
import msprime
import allel


@pytest.fixture(scope="session")
def example_data(tmp_path_factory):
    num_covariates = 3
    num_individuals = 5
    random_seed = 1024
    # use msprime to simulate some data
    ts = msprime.sim_ancestry(
        num_individuals,
        sequence_length=1e6,
        population_size=1e4,
        random_seed=random_seed,
        recombination_rate=1e-8,
    )
    ts = msprime.sim_mutations(
        ts, rate=1e-7, model=msprime.BinaryMutationModel(), random_seed=random_seed
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

    return (vcf_path, zarr_path, matrix_path, sample_data_path)

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======

>>>>>>> d904881 ( ran ruff formatting)
@pytest.fixture(scope="session")
def no_covariate_data(tmp_path_factory):
    num_individuals = 5
    random_seed = 2048
    ts = msprime.sim_ancestry(
        num_individuals,
        sequence_length=1e6,
        population_size=1e4,
        random_seed=random_seed,
        recombination_rate=1e-8,
    )
    ts = msprime.sim_mutations(
        ts, rate=1e-7, model=msprime.BinaryMutationModel(), random_seed=random_seed
    )
    matrix_path = tmp_path_factory.mktemp("nocov") / "test_matrix.tsv"
    sample_data_path = tmp_path_factory.mktemp("nocov") / "test_sample_data.tsv"
    sample_ids = [f"sample{i}" for i in range(ts.num_individuals)]
    haplo_matrix = ts.genotype_matrix().T
    diplo_matrix = haplo_matrix[::2] + haplo_matrix[1::2]
    df = pd.DataFrame(diplo_matrix, index=sample_ids)
    df.index.name = "sampleID"
    df.to_csv(matrix_path, sep="\t", index=True)
    rng = np.random.default_rng(random_seed)
    locs = rng.uniform(0, 1, size=(ts.num_individuals, 2))
    shuffled_indices = rng.permutation(ts.num_individuals)
    sample_data_df = pd.DataFrame(
        locs[shuffled_indices],
        index=[f"sample{i}" for i in shuffled_indices],
        columns=["x", "y"],
    )
    sample_data_df.index.name = "sampleID"
    sample_data_df.to_csv(sample_data_path, sep="\t", index=True)
    return (matrix_path, sample_data_path)

>>>>>>> 7bc048d (updates too allow location only inputs)

@pytest.fixture
def sample_locs():
    return np.array(
        [
            [1.0, 2.0, 10.0, 100.0, 5.0],
            [2.0, 3.0, 20.0, 200.0, 10.0],
            [3.0, 4.0, 30.0, 300.0, 15.0],
            [4.0, 5.0, 40.0, 400.0, 20.0],
            [5.0, 6.0, 50.0, 500.0, 25.0],
        ]
    )
