import pytest
import numpy as np
from pathlib import Path
from ecolocator.utils import load_genotypes

DATA_DIR = Path(__file__).parents[2] / "data"
VCF_PATH = DATA_DIR / "test_genotypes.vcf.gz"
ZARR_PATH = DATA_DIR / "test_genotypes.zarr"
MATRIX_PATH = DATA_DIR / "test_genotypes_matrix.tsv"

def test_load_genos_noinput():
    ''' testing load genotypes errors with no inputs '''
    with pytest.raises(ValueError, match="provide one of"):
        load_genotypes()

def test_load_genos_equivalence():
    ''' testing load genotypes returns same across data types '''
    genos_vcf, samples_vcf = load_genotypes(vcf_path=str(VCF_PATH))
    genos_zarr, samples_zarr = load_genotypes(zarr_path=str(ZARR_PATH))
    genos_matrix, samples_matrix = load_genotypes(matrix_path=str(MATRIX_PATH))

    # VCF and zarr are both phased — arrays should be exactly equal
    np.testing.assert_array_equal(genos_vcf, genos_zarr)
    np.testing.assert_array_equal(samples_vcf, samples_zarr)

    # Matrix stores dosage only (phase is lost) — compare alt allele counts
    np.testing.assert_array_equal(genos_vcf.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(genos_zarr.to_n_alt(), genos_matrix.to_n_alt())
    np.testing.assert_array_equal(samples_vcf, samples_matrix)
    np.testing.assert_array_equal(samples_zarr, samples_matrix)
