# Documentation deployment Makefile

# Variables
DOCS_BUILD_DIR ?= ../build/html
BUCKET_NAME := $(shell terraform output -raw bucket_name)
CLOUDFRONT_DOMAIN := $(shell terraform output -raw cloudfront_domain)
CLOUDFRONT_DISTRIBUTION_ID := $(shell terraform output -raw cloudfront_distribution_id)

.PHONY: plan apply deploy invalidate clean help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

plan: ## Show Terraform plan
	terraform plan

apply: ## Apply Terraform changes
	terraform apply

deploy: ## Deploy documentation to S3
	@if [ -z "$(BUCKET_NAME)" ]; then \
		echo "Error: Could not get bucket name. Make sure Terraform is initialized and applied."; \
		exit 1; \
	fi
	@echo "Deploying documentation to s3://$(BUCKET_NAME)/python/"
	aws s3 sync $(DOCS_BUILD_DIR) s3://$(BUCKET_NAME)/python/ --delete

invalidate: ## Invalidate CloudFront cache
	@if [ -z "$(CLOUDFRONT_DOMAIN)" ]; then \
		echo "Error: Could not get CloudFront domain. Make sure Terraform is initialized and applied."; \
		exit 1; \
	fi
	@echo "Invalidating CloudFront cache for $(CLOUDFRONT_DOMAIN)..."
	aws cloudfront --no-cli-pager create-invalidation --distribution-id $(CLOUDFRONT_DISTRIBUTION_ID) --paths "/python/*"

deploy-all: deploy invalidate ## Deploy docs and invalidate cache
	@echo "Deployment complete!"
