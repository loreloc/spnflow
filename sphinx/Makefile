SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SOURCEDIR     = .
GITHUBDIR     = ../docs
APISOURCE     = ../deeprob
BUILDDIR      = _build
SOURCESDIR    = _sources
APIDIR        = api

.PHONY: all clean

all: github

# Make the API pages
api:
	$(SPHINXAPI) -o $(APIDIR) $(APISOURCE)

# Make the HTML static site
html: api
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR)

# Prepare for Github pages
github: html
	cp -a "$(BUILDDIR)/html/." $(GITHUBDIR)

# Clean
clean:
	rm -rf $(BUILDDIR) $(APIDIR) $(GITHUBDIR)
