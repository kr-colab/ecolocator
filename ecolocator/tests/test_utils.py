import pytest
from ecolocator.utils import load_genotypes

def test_load_genos_noinput():
    ''' testing load genotypes errors with no inputs '''
    with pytest.raises(ValueError, match="provide one of"):
    	load_genotypes()

def test_load_genos_equivalence():
    ''' testing load genotypes returns same across data types '''
    assert False, "todo"
    #simulate some data (msprime maybe)
    #write out to zarr, vcf, matrix in temp dir
    #read in each of above
    #use numpy.testing.array_equals to test that above are same
