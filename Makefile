.PHONY: test

test:
	pytest -k "not integration" --cov=src/ --cov-report=xml tests/
