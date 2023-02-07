### How to run the tests.

Go to the root, then run:

```bash
python -m pytest -s
```

To run with a specific marker, do the following.

```bash
python -m pytest -s -m meta_evaluation
```

Or with coverage.
```bash
pytest tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov=metaquantus
```