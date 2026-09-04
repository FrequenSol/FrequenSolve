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
	python scripts/check_optional_extra_contracts.py \
		--run visual \
		--coverage-output tests/output/optional-visual-coverage.json
	python scripts/check_optional_extra_contracts.py \
		--run seismic-io \
		--coverage-output tests/output/optional-seismic-io-coverage.json

.PHONY: validate-optional-extra-contracts
validate-optional-extra-contracts:
	python scripts/check_optional_extra_contracts.py --validate

.PHONY: test-optional-extra-contract
test-optional-extra-contract:
	@test -n "$(EXTRA)" || (echo "EXTRA is required" >&2; exit 2)
	python scripts/check_optional_extra_contracts.py \
		--run "$(EXTRA)" \
		--coverage-output "tests/output/optional-$(EXTRA)-coverage.json"

.PHONY: test-hpc-hermetic
test-hpc-hermetic:
	python -m pytest \
	-ra \
	-o addopts='' \
	--strict-markers \
	-m hpc_hermetic \
	--cov=src/frequensolve/orchestrator/sites/hpc \
	--cov-config=tests/hpc.coveragerc \
	--cov-branch \
	--cov-report=term \
	--cov-fail-under=67.5 \
	tests/

.PHONY: typecheck
typecheck:
	python scripts/check_mypy_baseline.py
