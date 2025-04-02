FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for weasyprint
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-cffi \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file first (if it exists)
COPY .env* ./ 

# Copy the entire application
COPY . .

# Create directory for output files and ensure correct permissions
RUN mkdir -p eda_output/figures eda_output/interactive \
    && chmod -R 777 eda_output

# Make the entry script executable
RUN chmod +x docker-entrypoint.sh

# Expose the port Azure App Service expects
EXPOSE 8000

# Set environment variable to ensure Streamlit runs properly in Docker
ENV PYTHONUNBUFFERED=1

# Use the entry script to start the application
CMD ["/app/docker-entrypoint.sh"]
