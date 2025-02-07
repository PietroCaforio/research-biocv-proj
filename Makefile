.PHONY: deps data

deps:
	@echo "Installing dependencies..."
	conda env create -f conda_requirements.yml
	conda activate research_biocv_proj
	pre-commit install
