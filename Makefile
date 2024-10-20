.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

# MODULE1=$(CURRENT_DIR)/module-1
# MODULE1_CRITICAL_THINKING=$(MODULE1)/critical-thinking

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

# .PHONY: m1
# m1: ## executes module 1 critical thinking
# 	@echo "executing module 1 critical thinking ..."
# 	@cd $(MODULE1_CRITICAL_THINKING) && ./machine-info.sh
# 	@echo "completed module 1 critical thinking."

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@pip install pandas numpy nltk scikit-learn
	@unzip $(PP_DATA)/$(PP_TRAINING_TEST_DATA) -d $(PP_DATA) && chmod 644 -R $(PP_DATA)

.PHONY: pp
pp: ## executes portfolio project
	@echo "pp: starting portfolio project"
	@$(PP)/$(PP_APP)
	@echo "pp: completed portfolio project"

.PHONY: setup-python

setup-os: ## setup os dependencies
	@echo "installing os tools"
	@sudo apt update && sudo apt upgrade -y && sudo apt install -y python3.10-venv make

.PHONY: setup-cuda-toolkit
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
setup-cuda-toolkit: ## setup cuda toolkit
	@echo "installing cuda toolkit 12.6.1"
	@wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run && \
		sudo sh cuda_12.6.1_560.35.03_linux.run; \
		rm cuda_12.6.1_560.35.03_linux.run
	@nvcc --version

.PHONY: setup-cudnn
# https://developer.nvidia.com/rdp/cudnn-archive
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
