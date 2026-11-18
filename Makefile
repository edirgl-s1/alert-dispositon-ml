.PHONY: setup lint fmt typecheck test train data mlflow notebook clean
setup:      ## install deps + git hooks
	uv sync --all-groups && uv run pre-commit install
lint:
	uv run ruff check . && uv run ruff format --check .
fmt:
	uv run ruff check --fix . && uv run ruff format .
typecheck:
	uv run mypy
test:
	uv run pytest
data:
	uv run alertml prepare-data
train:      ## make train ARGS="-o model.name=logreg"
	uv run alertml train -c configs/default.yaml -c configs/synthetic.yaml $(ARGS)
mlflow:     ## experiment tracking UI at :5000
	uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
notebook:
	uv run --group notebook jupyter lab
