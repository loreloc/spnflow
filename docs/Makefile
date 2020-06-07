SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SOURCEDIR     = .
APISOURCE     = ../spnflow
BUILDDIR      = _build
APIDIR        = api

.PHONY: all clean

# Build all
all: api html github

# Make the API pages
api:
	$(SPHINXAPI) -o "$(APIDIR)" "$(APISOURCE)"

# Make the HTML static site
html: api
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Make the Github Pages stsatic site
github: html
	cp -a "$(BUILDDIR)/html/." "$(SOURCEDIR)"
	cp "$(SOURCEDIR)/../README.md" "$(SOURCEDIR)"

# Clean
clean:
	rm -rf "$(BUILDDIR)" "$(APIDIR)"
