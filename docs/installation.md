# Installation

`ecoLocator` is managed with [uv](https://docs.astral.sh/uv/) and requires
Python 3.11 or newer.

## From source

The package lives in the `ecolocator/` subdirectory of the repository. Clone the
repository and sync the environment:

```bash
git clone https://github.com/kr-colab/ecolocator.git
cd ecolocator/ecolocator
uv sync
```

This creates a `.venv` with all runtime dependencies. Run commands inside it with
`uv run`, for example:

```bash
uv run ecolocator --help
```

## Development environment

To also install the development tools (test suite, linter) and the documentation
toolchain, sync the corresponding dependency groups:

```bash
uv sync --group dev --group docs
```

Then run the tests and linter:

```bash
uv run pytest
uv run ruff check
```
