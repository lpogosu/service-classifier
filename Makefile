.DEFAULT_GOAL := help
PYTHON ?= python
DATASET ?= data/synthetic_dataset.csv
ROWS ?= 400

.PHONY: help install data train run api test lint typecheck eval docker clean

help: ## Показать список целей
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Установить зависимости вместе с dev-инструментами
	$(PYTHON) -m pip install -e ".[dev]"

data: ## Сгенерировать синтетический датасет
	$(PYTHON) -m data.synthetic --rows $(ROWS) --output $(DATASET)

train: data ## Обучить классификатор стадии 2 на датасете
	$(PYTHON) -m training.train_split_classifier --dataset $(DATASET)

run: data ## Прогнать пайплайн по датасету и посчитать метрики
	$(PYTHON) -m service_classifier --dataset $(DATASET) --evaluate \
		--output results/predictions.json

eval: run ## Синоним run: прогон с метриками

api: ## Поднять HTTP-сервер на localhost:8000
	$(PYTHON) -m uvicorn api.server:app --host 0.0.0.0 --port 8000

test: ## Запустить тесты
	$(PYTHON) -m pytest -q

lint: ## Проверить стиль
	$(PYTHON) -m ruff check .

typecheck: ## Проверить типы
	$(PYTHON) -m mypy

docker: ## Собрать и запустить контейнер
	docker compose up --build -d

clean: ## Убрать артефакты сборки и запуска
	rm -rf results .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
