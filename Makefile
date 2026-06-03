.PHONY: install lint lint-fix format test run mlflow mlflow-ui clean ALL

POETRY = poetry
RUFF   = $(POETRY) run ruff
PYTEST = $(POETRY) run pytest

ALL: lint test

install:
	$(POETRY) install

lint:
	@echo "=== Executando o Linter e Formatador (Ruff) ==="
	$(RUFF) check src/
	$(RUFF) format --check src/

lint-fix:
	@echo "=== Corrigindo automaticamente (Ruff) ==="
	$(RUFF) check --fix src/
	$(RUFF) format src/

test:
	@echo "=== Executando a Suite de Testes (Pytest) ==="
	$(PYTEST) src/tests/ -v --tb=short

run:
	@echo "=== Iniciando o Pipeline ==="
	$(POETRY) run python -c "from src.config import setup_logging; setup_logging(); from src.pipeline import ChurnPipeline; ChurnPipeline().run()"

MLFLOW_EXE ?= $(shell $(POETRY) env info --path 2>/dev/null)/Scripts/mlflow.exe

mlflow:
	$(MLFLOW_EXE) ui --backend-store-uri sqlite:///mlruns.db --port 5000

mlflow-ui:
	@echo "=== Iniciando a interface do MLflow ==="
	$(POETRY) run mlflow ui $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
