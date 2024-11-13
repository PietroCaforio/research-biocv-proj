.PHONY: deps data

deps:
	@echo "Installing dependencies..."
	conda env create -f conda_requirements.yml
	pre-commit install
