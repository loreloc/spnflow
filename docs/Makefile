SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SOURCEDIR     = .
APISOURCE     = ../spnflow
BUILDDIR      = _build
APIDIR        = api
README        = README.*

.PHONY: all clean

# Build all
all: api html

# Make the API pages
api:
	$(SPHINXAPI) -o "$(APIDIR)" "$(APISOURCE)"

# Make the HTML static site
html: api
	cp "$(SOURCEDIR)/../README.md" "$(SOURCEDIR)"
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	cp -a "$(BUILDDIR)/html/." "$(SOURCEDIR)"

# Clean
clean:
	rm -rf "$(BUILDDIR)" "$(APIDIR)" "$(README)"
