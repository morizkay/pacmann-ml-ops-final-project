.PHONY: setup data train evaluate run-api test docker-build docker-run clean mlflow-up mlflow-down pipeline status logs

# Pipeline Commands
pipeline: data train evaluate

# Setup
setup:
	pip install -r requirements.txt
	pip install -e .

# Data Pipeline
data:
	python src/data/load_data.py
	python src/features/preprocessing.py

# Model Training
train:
	python src/models/train.py

# Model Evaluation
evaluate:
	python src/evaluation/evaluate.py

# Run API
run-api:
	python src/api/app.py

# Run Tests
test:
	pip install -e . && PYTHONPATH=. pytest tests/

# Docker Commands
docker-build:
	docker build -t gk-prediction .

docker-run:
	docker run -p 8000:8000 gk-prediction

# MLflow Services
mlflow-up:
	docker-compose -f docker/mlflow/docker-compose.yml up -d

mlflow-down:
	docker-compose -f docker/mlflow/docker-compose.yml down

mlflow-logs:
	docker-compose -f docker/mlflow/docker-compose.yml logs -f

# Development Environment
dev-up:
	docker-compose -f docker/dev/docker-compose.yml up -d

dev-down:
	docker-compose -f docker/dev/docker-compose.yml down

dev-logs:
	docker-compose -f docker/dev/docker-compose.yml logs -f

# Staging Environment
staging-up:
	docker-compose -f docker/staging/docker-compose.yml up -d

staging-down:
	docker-compose -f docker/staging/docker-compose.yml down

staging-logs:
	docker-compose -f docker/staging/docker-compose.yml logs -f

# Production Environment
prod-up:
	docker-compose -f docker/prod/docker-compose.yml up -d

prod-down:
	docker-compose -f docker/prod/docker-compose.yml down

prod-logs:
	docker-compose -f docker/prod/docker-compose.yml logs -f

# Status Commands
status:
	@echo "=== MLflow Services ==="
	@docker-compose -f docker/mlflow/docker-compose.yml ps
	@echo "\n=== Development Services ==="
	@docker-compose -f docker/dev/docker-compose.yml ps
	@echo "\n=== Staging Services ==="
	@docker-compose -f docker/staging/docker-compose.yml ps
	@echo "\n=== Production Services ==="
	@docker-compose -f docker/prod/docker-compose.yml ps

# Clean
clean:
	rm -rf data
	rm -rf models
	rm -rf metrics
	rm -rf gk_prediction.egg-info
	rm -rf mlruns
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "Available commands:"
	@echo "  make pipeline     - Run full ML pipeline (data, train, evaluate)"
	@echo "  make setup        - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make status       - Show status of all services"
	@echo "  make *-up         - Start services (mlflow/dev/staging/prod)"
	@echo "  make *-down       - Stop services"
	@echo "  make *-logs       - View service logs"
	@echo "  make clean        - Clean generated files"