.PHONY: generate_reference_images
generate_reference_images:
	python -m pytest -ra -m "visual and not integration and not cloud and not hpc and not interactive" --mpl-generate-path=tests/reference_images/ tests/

.PHONY: test
test:
	python -m pytest \
	-ra \
	-m "not integration and not cloud and not hpc and not interactive and not visual" \
	--cov=src/ --cov-report=xml \
	--mpl --mpl-baseline-path=tests/reference_images/ --mpl-generate-summary=html --mpl-results-path=tests/output/ \
	tests/
