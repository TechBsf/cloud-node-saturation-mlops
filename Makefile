.PHONY: install dev-install lint typecheck test train api

install:
	python -m pip install --upgrade pip
	python -m pip install .

dev-install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest -v

train:
	python scripts/train_with_mlflow.py

api:
	uvicorn scripts.app:app --host 0.0.0.0 --port 8000
