from ecolocator import EcoLocator


def test_ecolocator_init_defaults():
    """EcoLocator initializes with correct default hyperparameters"""
    model = EcoLocator()
    assert model.nlayers == 10
    assert model.width == 256
    assert model.dropout_prop == 0.25
    assert model.loc_weight == 1.0
    assert model.env_weight == 1.0
    assert model.cov_transforms is None


def test_ecolocator_init_custom():
    """EcoLocator stores custom hyperparameters correctly"""
    model = EcoLocator(nlayers=5, width=128, cov_transforms=['none', 'log'])
    assert model.nlayers == 5
    assert model.width == 128
    assert model.cov_transforms == ['none', 'log']
