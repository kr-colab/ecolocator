import numpy as np
import logging
from typing import Self


class EcoLocator:
    """
    Docstringer describing class
    """

    def __init__(
        self,
        project_path: str = None,  # path to project directory
    ):
        self.project_path = project_path
        logging.info(f"Using project path {self.project_path}")

    def get_project_path(self) -> str:
        return self.project_path

    @staticmethod
    def load_from_yaml(
        yaml_path: str,
    ) -> Self:
        assert False, "NOT IMPLEMENTED YET"
