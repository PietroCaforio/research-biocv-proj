.PHONY: deps data

deps:
	@echo "Installing dependencies..."
	conda env create -f conda_requirements.yml
	conda activate research_biocv_proj
	pre-commit install
backbone-pretrained-weights:
	@echo "Downloading Resnet from https://github.com/kenshohara/3D-ResNets-PyTorch"
	gdown --id 1fFN5J2He6eTqMPRl_M9gFtFfpUmhtQc9 -O ./models/pretrain_weights/
nbia-data-retriever:
	@echo "Installing nbia-data-retriever cli"
	cd ~/
	wget https://github.com/CBIIT/NBIA-TCIA/releases/download/DR-4_4_3-TCIA-20240916-1/nbia-data-retriever_4.4.3-1_amd64.deb
	ar x nbia-data-retriever_4.4.3-1_amd64.deb
	tar xvf control.tar.xz
	tar xvf data.tar.xz
CPTAC-PDA-CT:
	@echo "Downloading CPTAC-PDA radiology data"
	~/opt/nbia-data-retriever/bin/nbia-data-retriever --cli data/raw/CPTAC_PDA_77/CT.tcia -d data/raw/CPTAC_PDA_77/
CPTAC-UCEC-CT:
	@echo "Downloading CPTAC-PDA radiology data"
	~/opt/nbia-data-retriever/bin/nbia-data-retriever --cli data/raw/69PatientsCPTACUCEC/manifest-1728901427271.tcia -d data/raw/69PatientsCPTACUCEC/
