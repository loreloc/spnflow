PYTHON = python
SETUP_SRC = setup.py

.PHONY: all clean

all: pip_make pip_upload

pip_make: $(SETUP_SRC)
	$(PYTHON) $< sdist bdist_wheel

pip_upload:
	$(PYTHON) -m twine upload dist/*

clean:
	rm -rf dist
	rm -rf build
	rm -rf deeprob.egg-info
