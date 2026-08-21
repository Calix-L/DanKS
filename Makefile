PYTHON ?= python3

.PHONY: help overview show test audit package clean-dist

help: ## Show available repository commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

overview: ## List V1, V2, and V3
	$(PYTHON) -m danks_repo list

show: ## Show one generation (for example: make show GENERATION=v3)
	$(PYTHON) -m danks_repo show $(GENERATION)

test: ## Run the repository test suite
	$(PYTHON) -m pytest -q

audit: ## Verify layout, source boundaries, and generation manifests
	$(PYTHON) -m danks_repo verify

package: audit ## Build all deterministic source archives
	$(PYTHON) -m danks_repo package --generation all

clean-dist: ## Remove only generated archives below ./dist
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path.cwd()/'dist'; shutil.rmtree(p) if p.is_dir() else None"
