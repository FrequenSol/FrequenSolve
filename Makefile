.PHONY: generate_reference_images
generate_reference_images:
	python -m pytest -ra -m "visual and not integration and not cloud and not hpc and not interactive" --mpl-generate-path=tests/reference_images/ tests/test_ex01_simple.py

.PHONY: test
test:
	python -m pytest \
	-ra \
	-m "not integration and not cloud and not hpc and not interactive and not visual" \
	--cov=src/ --cov-report=term --cov-report=xml --cov-report=json:tests/output/coverage.json \
	tests/
	python scripts/check_coverage_thresholds.py tests/output/coverage.json

.PHONY: test-property-contracts
test-property-contracts:
	PYTHONPATH="$(CURDIR)/src" \
	FREQUENSOLVE_HYPOTHESIS_PROFILE=pr python -m pytest \
		-ra \
		-o addopts='' \
		--strict-markers \
		-m property_contract \
		tests/

.PHONY: test-property-campaign
test-property-campaign:
	python scripts/run_property_campaign.py

.PHONY: test-optional-extras
test-optional-extras:
	MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true python -m pytest \
	-ra \
	-o addopts='' \
	--strict-markers \
	-m "not integration and not cloud and not hpc and not interactive" \
	--mpl --mpl-baseline-path=tests/reference_images/ --mpl-generate-summary=html --mpl-results-path=tests/output/ \
	tests/test_ex01_simple.py tests/test_seismic_plotting.py tests/test_trace_record.py

.PHONY: typecheck
typecheck:
	python -m mypy
