import numpy as np
from ecolocator.network import build_network, build_callbacks

def test_build_network_dual_output():
    """testing that build_network returns a model with exactly two output heads"""
    model = build_network(n_snps=100, num_covs=3)
    assert len(model.outputs) == 2

def test_build_network_output_shape():
    """testing that build_network loc head outputs (n_samples, 2) and env head outputs (n_samples, num_covs)"""
    model = build_network(n_snps=100, num_covs=3)
    input = np.random.rand(5, 100)
    loc_out, env_out = model.predict(input, verbose=0)
    assert loc_out.shape == (5, 2)
    assert env_out.shape == (5, 3)

def test_build_callbacks_returns_two():
    """testing that build_callbacks returns a list of two callbacks"""
    callbacks = build_callbacks()
    assert len(callbacks) == 2

def test_build_callbacks_earlystopping_patience():
    """testing that patience parameter is correctly passed to EarlyStopping callback"""
    callbacks = build_callbacks(patience=42)
    earlystop = callbacks[0]
    assert earlystop.patience == 42

def test_build_network_forward_pass():
    """testing that build_network produces a model that runs a forward pass without error"""
    model = build_network(n_snps=50, num_covs=2)
    input = np.random.rand(3, 50)
    loc_out, env_out = model.predict(input, verbose=0)
    assert loc_out is not None
    assert env_out is not None