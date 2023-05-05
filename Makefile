# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Download data
download_data:
	python data_download.py

# Generate features
features:
	python feature_engineering.py

# Train the model
train:
	python model_train.py

# Evaluate the model
evaluate:
	python model_evaluation.py

# Deploy the model
deploy:
	python app.py

# Test the model
test:
	python test.py

# Clean up intermediate files
clean:
	rm -rf data/processed/
	rm -rf models/

# Run all steps
all: install download_data features train evaluate deploy test