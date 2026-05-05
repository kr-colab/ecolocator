from ecolocator import EcoLocator


def test_ecolocator_init():
    project_path = "test_project"
    ecolocator = EcoLocator(project_path=project_path)
    assert ecolocator.get_project_path() == project_path


def test_ecolocator_load_from_yaml():
    yaml_path = "test_project.yaml"
    ecolocator = EcoLocator.load_from_yaml(yaml_path=yaml_path)
