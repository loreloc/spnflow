SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SOURCEDIR     = .
APISOURCE     = ../spnflow
BUILDDIR      = _build
APIDIR        = api

.PHONY: all clean

# Build all
all: api html

# Make the API pages
api:
	$(SPHINXAPI) -o "$(APIDIR)" "$(APISOURCE)"

# Make the HTML static site
html: api
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Clean
clean:
	rm -rf "$(BUILDDIR)" "$(APIDIR)"
