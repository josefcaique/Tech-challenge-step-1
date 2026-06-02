.PHONY: lint format test run mlflow mlflow-ui ALL

POETRY = poetry
RUFF = $(POETRY) run ruff
PYTEST = $(POETRY) run pytest
MLFLOW = $(POETRY) run mlflow

ALL: lint test

lint:
	@echo "=== Executando o Linter e Formatador (Ruff) ==="
	$(RUFF) check . --fix
	$(RUFF) format .

test:
	@echo "=== Executando a Suite de Testes (Pytest) ==="
	$(PYTEST)

run:
	@echo "=== Iniciando a Aplicação ==="
	$(POETRY) run python src/main.py

mlflow:
	@echo "=== Executando o MLflow no ambiente do Poetry ==="
	$(MLFLOW) $(ARGS)

mlflow-ui:
	@echo "=== Iniciando a interface do MLflow no ambiente do Poetry ==="
	$(MLFLOW) ui $(ARGS)