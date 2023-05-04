# Install dependencies
install:
    pip install -r requirements.txt

# Download data
download_data:
    python download_data.py

# Generate features
features:
    python feature_engineering.py

# Train the model
train:
    python model_training.py

# Evaluate the model
evaluate:
    python model_evaluation.py

# Deploy the model
deploy:
    python deploy_model.py

# Clean up intermediate files
clean:
    rm -rf data/processed/
    rm -rf models/

# Run all steps
all: install download_data features train evaluate deploy
