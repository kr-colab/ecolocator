# Minimal makefile for Sphinx documentation.
# Build from the repo root with the package env, e.g.:
#   uv run --project ../ecolocator make -C docs html

SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route unknown targets to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
