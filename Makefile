PYTHON := uv run python

.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --extra dev
	pre-commit install

.PHONY: type
type:
	SKIP=no-commit-to-branch pre-commit run -a pyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	SKIP=no-commit-to-branch pre-commit run -a ruff-lint
	SKIP=no-commit-to-branch pre-commit run -a ruff-format

.PHONY: check
check:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

.PHONY: test
test:
	$(PYTHON) -m pytest tests/

.PHONY: test-all
test-all:
	$(PYTHON) -m pytest tests/ --runslow

.PHONY: exp-tms-train
exp-tms-train:
	$(PYTHON) spd/experiments/tms/train_tms.py

.PHONY: exp-tms-decomp
exp-tms-decomp:
	$(PYTHON) spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_config.yaml

.PHONY: exp-tms
exp-tms: exp-tms-train exp-tms-decomp
	@echo "train and then decompose TMS model"

.PHONY: exp-lm
exp-lm:
	$(PYTHON) spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ts_config.yaml