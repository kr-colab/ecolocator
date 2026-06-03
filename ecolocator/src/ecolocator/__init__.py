from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ecolocator_class import EcoLocator as EcoLocator

__all__ = ["EcoLocator"]


def __getattr__(name: str):
    # Import lazily so that merely importing the package (e.g. to resolve the
    # CLI entry point for `--help`) does not pull in TensorFlow.
    if name == "EcoLocator":
        from .ecolocator_class import EcoLocator

        return EcoLocator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
