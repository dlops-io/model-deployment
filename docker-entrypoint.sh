#!/bin/bash

echo "Container is running!!!"
echo "Architecture: $(uname -m)"

# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
# Set GCP Project Details
gcloud config set project "$GCP_PROJECT"

# The virtual environment (/.venv) is already on PATH (set in the Dockerfile)
echo "Environment ready!"
echo "Python: $(which python) ($(python --version))"
echo "UV version: $(uv --version)"

# Keep a shell open
exec /bin/bash
