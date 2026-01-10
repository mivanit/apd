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

.PHONY: format-m
format-m:
	$(PYTHON) -m ruff check --fix spd/analysis
	$(PYTHON) -m ruff check --fix notebooks/mivanit
	$(PYTHON) -m ruff format spd/analysis
	$(PYTHON) -m ruff format notebooks/mivanit

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

.PHONY: exp-mlp-train
exp-mlp-train:
	$(PYTHON) spd/experiments/resid_mlp/train_resid_mlp.py

.PHONY: exp-mlp-decomp
exp-mlp-decomp:
	$(PYTHON) spd/experiments/resid_mlp/resid_mlp_decomposition.py spd/experiments/resid_mlp/resid_mlp_config.yaml

.PHONY: exp-mlp
exp-mlp: exp-mlp-train exp-mlp-decomp
	@echo "train and then decompose MLP model"

.PHONY: exp-tms
exp-tms: exp-tms-train exp-tms-decomp
	@echo "train and then decompose TMS model"

.PHONY: exp-lm
exp-lm:
	$(PYTHON) spd/experiments/lm/lm_decomposition.py spd/experiments/lm/ts_config.yaml --weights-only=False