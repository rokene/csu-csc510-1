.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

# MODULE1=$(CURRENT_DIR)/module-1
# MODULE1_CRITICAL_THINKING=$(MODULE1)/critical-thinking

PP=$(CURRENT_DIR)/portfolio-project
PP_DATA=$(PP)/data

PP_APP=app.py
PP_TRAINING_TEST_DATA=trainingandtestdata.zip

SIMPLE_ANN=simple-ann

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

.PHONY: setup-simple-ann
setup-simple-ann: ## setup simple ann project
	@cd $(SIMPLE_ANN) && python -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install tensorflow==2.12.0 && \
		./tf-test.py
