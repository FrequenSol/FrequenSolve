.PHONY: generate_reference_images
generate_reference_images:
	poetry run pytest -ra -k "not integration" --mpl-generate-path=tests/reference_images/ tests/

.PHONY: test
test:
	poetry run pytest \
	-ra -k "not integration" \
	--cov=src/ --cov-report=xml \
	--mpl --mpl-baseline-path=tests/reference_images/ --mpl-generate-summary=html \
	tests/
