echo "spnflow testing..."
export PYTHONPATH=.
find tests/test_*.py -print0 | xargs -n 1 -0 python3 -m pytest -s
