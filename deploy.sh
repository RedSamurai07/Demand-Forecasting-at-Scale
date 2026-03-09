#!/bin/bash

# Deployment script for EC2
echo "Starting deployment..."

# 1. Update system
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# 2. Pull latest code (assuming git is set up)
# git pull origin main

# 3. Build Docker image
docker build -t demand-forecaster .

# 4. Stop any existing container
docker stop forecaster-app || true
docker rm forecaster-app || true

# 5. Run the container
docker run -d --name forecaster-app demand-forecaster

echo "Deployment finished successfully!"
