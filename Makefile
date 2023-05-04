# Define variables
PROJECT_NAME = iris_classification
PYTHON_INTERPRETER = python3

# Define targets
.PHONY: data
data:
    $(PYTHON_INTERPRETER) data_download.py

.PHONY: features
features:
    $(PYTHON_INTERPRETER) feature_engineering.py

.PHONY: train
train:
    $(PYTHON_INTERPRETER) model_training.py

.PHONY: evaluate
evaluate:
    $(PYTHON_INTERPRETER) model_evaluation.py

.PHONY: deploy
deploy:
    $(PYTHON_INTERPRETER) app.py
