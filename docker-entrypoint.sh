#!/bin/bash

# This script serves as the entry point for the Docker container

echo "Starting Streamlit EDA Tool..."

# Print Python and pip versions for debugging
python --version
pip --version

# Check for .env file and load environment variables if present
if [ -f /app/.env ]; then
    echo "Found .env file, loading environment variables..."
    export $(grep -v '^#' /app/.env | xargs)
else
    echo "Warning: No .env file found"
fi

# Print environment variables (securely)
echo "Checking environment variables..."
env_vars=("AZURE_OPENAI_ENDPOINT" "AZURE_OPENAI_DEPLOYMENT" "AZURE_API_VERSION" "EDA_DEPTH")
for var in "${env_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var is set"
    else
        echo "❌ $var is NOT set"
    fi
done
# Check API key securely without printing it
if [ -n "$AZURE_OPENAI_API_KEY" ]; then
    echo "✅ AZURE_OPENAI_API_KEY is set"
else
    echo "❌ AZURE_OPENAI_API_KEY is NOT set"
fi

# Check if app.py exists
if [ ! -f /app/app.py ]; then
    echo "ERROR: app.py not found in /app directory!"
    echo "Contents of /app directory:"
    ls -la /app
    exit 1
fi

# Create clean output directory
echo "Preparing output directory..."
mkdir -p /app/eda_output/figures
mkdir -p /app/eda_output/interactive
# Clear any existing files in the directories if they exist
rm -f /app/eda_output/*.md /app/eda_output/*.html /app/eda_output/*.json /app/eda_output/*.xlsx /app/eda_output/*.zip
rm -f /app/eda_output/figures/*
rm -f /app/eda_output/interactive/*

# Print directory structure for debugging
echo "Directory structure:"
find /app -type f -name "*.py" | sort

# Start the Streamlit app with proper configurations
echo "Starting Streamlit app..."
streamlit run --server.port=8000 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false /app/app.py
