.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project
PP_DATA=$(PP)/data

PP_APP=app.py
PP_TRAINING_TEST_DATA=trainingandtestdata.zip

SIMPLE_ANN=simple-ann

INFORMED_SEARCH=informed-search

NAIVE_BAYES=naive-bayes-classifier

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(PP) && \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	@echo "pp: setting up portfolio project data"
	@git lfs install
	@git pull
	@git lfs pull
	@unzip $(PP_DATA)/$(PP_TRAINING_TEST_DATA) -d $(PP_DATA) && \
		sudo chown -R rokene:rokene $(PP_DATA) && \
		sudo chmod -R u+rwX $(PP_DATA)

.PHONY: pp-test
pp-test: ## executes test portfolio project
	@cd $(PP) && \
		. venv/bin/activate && \
		nvidia-smi && \
		nvcc --version

.PHONY: pp
pp: ## executes portfolio project
	@echo "checking if gpu libs are available"; make pp-test
	@echo "pp: starting portfolio project"
	@cd $(PP) && \
		. venv/bin/activate && \
		$(PP)/$(PP_APP)
	@echo "pp: completed portfolio project"

.PHONY: setup-os
setup-os: ## setup os dependencies
	@echo "installing os tools"
	@sudo apt update && sudo apt upgrade -y && sudo apt install -y python3.10-venv make

# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
.PHONY: setup-cuda-toolkit
setup-cuda-toolkit: ## setup cuda toolkit
	@echo "installing cuda toolkit 12.6.1"
	@wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run && \
		sudo sh cuda_12.6.1_560.35.03_linux.run; \
		rm cuda_12.6.1_560.35.03_linux.run
	@nvcc --version

# https://developer.nvidia.com/rdp/cudnn-archive
.PHONY: setup-cudnn
setup-cudnn: ## setup cudnn
	@echo "installing cudnn-linux-x86_64-8.9.7.29_cuda12"
	@sudo apt install -y libcusolver-12 libcusparse-12 libcublas-12
	@cd supporting-artifacts && \
		tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz && \
		cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive && \
		sudo cp include/cudnn*.h /usr/local/cuda-12.6/include/ && \
		sudo cp -P lib/libcudnn* /usr/local/cuda-12.6/lib64/ && \
		sudo chmod a+r /usr/local/cuda-12.6/include/cudnn*.h /usr/local/cuda-12.6/lib64/libcudnn*
	@cat /usr/local/cuda-12.6/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

.PHONY: setup-simple-ann
setup-simple-ann: ## setup simple ann project
	@cd $(SIMPLE_ANN) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt && \
		./tf-test.py

.PHONY: simple-ann
simple-ann: ## executes simple ann
	@cd $(SIMPLE_ANN) && \
		. venv/bin/activate && \
		./simple-ann-numpy.py

.PHONY: towers-hanoi
towers-hanoi: ## executes informed search towers of hanoi
	@cd $(INFORMED_SEARCH) && \
		./towers-hanoi.py

.PHONY: setup-naive-bayes-classifier
setup-naive-bayes-classifier: ## setup the naive bayes project
	@cd $(NAIVE_BAYES) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

.PHONY: naive-bayes-classifier
naive-bayes-classifier: ## executes naive-bayes-classifier
	@cd $(NAIVE_BAYES) && \
		. venv/bin/activate && \
		./classifier.py
