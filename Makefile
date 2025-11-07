PYTHONWARNINGS="ignore:NotOpenSSLWarning" source venv/bin/activate && python src/train.pyi.PHONY: setup train run docker-build docker-run clean help

setup:
	@echo "🔧 Setting up virtual environment..."
	python3 -m venv venv
	@echo "📦 Installing dependencies..."
	source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Setup complete."

# 🔍 Preflight check to verify environment setup
preflight:
	@echo "🧪 Running preflight checks..."
	@python3 --version
	@which python3
	@which pip
	@echo "✅ Python and pip detected."
	@pip show mlflow >/dev/null 2>&1 && echo "✅ MLflow installed." || echo "⚠️ MLflow not found."
	@pip show dvc >/dev/null 2>&1 && echo "✅ DVC installed." || echo "⚠️ DVC not found."
	@pip show fastapi >/dev/null 2>&1 && echo "✅ FastAPI installed." || echo "⚠️ FastAPI not found."
	@pip show uvicorn >/dev/null 2>&1 && echo "✅ Uvicorn installed." || echo "⚠️ Uvicorn not found."
	@docker --version >/dev/null 2>&1 && echo "✅ Docker installed." || echo "⚠️ Docker not found."
	@echo "✅ Preflight checks complete."

train:
	@echo "🚀 Training model..."
	PYTHONWARNINGS="ignore:NotOpenSSLWarning" source venv/bin/activate && python src/train.py
	@echo "✅ Training complete."

run:
	@echo "🏃 Running FastAPI app..."
	# Activate venv and run uvicorn in one command
	@. venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t mlops-fasttrack:latest .

docker-run:
	@echo "🚀 Running Docker container..."
	docker run -d -p 8000:8000 mlops-fasttrack:latest

clean:
	@echo "🧹 Cleaning up temporary files..."
	rm -rf __pycache__ venv mlruns .dvc data/processed
	@echo "✅ Clean complete."

help:
	@echo "Available commands:"
	@echo "  make setup         - Create venv and install dependencies"
	@echo "  make train         - Run training script"
	@echo "  make run           - Start FastAPI app locally"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run app in Docker container"
	@echo "  make clean         - Remove generated files"
