SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SOURCEDIR     = .
APISOURCE     = ../spnflow
BUILDDIR      = _build
SOURCESDIR    = _sources
IMAGESDIR     = _images
APIDIR        = api
OTHERS        = .buildinfo .nojekyll objects.inv *.html *.js

.PHONY: all clean

# Build all
all: html

# Make the API pages
api:
	$(SPHINXAPI) -o $(APIDIR) $(APISOURCE)

# Make the HTML static site
html: api
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)
	cp -a "$(BUILDDIR)/html/." $(SOURCEDIR)

# Clean
clean:
	rm -rf $(BUILDDIR) $(SOURCESDIR) $(IMAGESDIR) $(APIDIR) $(OTHERS)
