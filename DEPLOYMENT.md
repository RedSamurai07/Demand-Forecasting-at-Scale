# 🚀 AWS EC2 Production Deployment Guide

This guide details the infrastructure configuration, container deployment strategy, and validation steps to host the Fraud Detection Anomaly Analysis Service on an AWS EC2 cloud instance using Docker.

---

## System Architecture Endpoints
* **Core API Layer (FastAPI):** `http://<EC2_PUBLIC_IP>:5000`
* **API Interactive Playground (Swagger UI):** `http://<EC2_PUBLIC_IP>:5000/docs`
* **AI-Powered Fraud Detection Interface (Gradio UI):** `http://<EC2_PUBLIC_IP>:7860`

---

## Step 1: Launch and Configure EC2 Instance

1. **Provision Compute:** Launch an Amazon EC2 instance using **Ubuntu 22.04 LTS** (a `t2.micro` or `t3.small` instance is recommended).
2. **Configure Firewall / Security Groups:** Expose the minimum necessary ingress ports to authorize external traffic pipelines securely:
   * **SSH (Port 22):** For secure remote shell administration.
   * **FastAPI (Port 5000):** To handle client requests and fraud prediction inference payloads.
   * **Gradio UI (Port 7860):** To serve the interactive AI-powered fraud detection web interface.

### Establish Secure SSH Connection

#### On Windows (PowerShell):
```powershell
# Restrict file permissions to the current user (Windows equivalent of chmod 400)
icacls "fraud-detect-key.pem" /inheritance:r
icacls "fraud-detect-key.pem" /grant:r "${env:USERNAME}:R"

# Connect to the remote instance
ssh -i "fraud-detect-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

#### On Linux/Mac:
```bash
# Set strict read-only permissions for the private key
chmod 400 fraud-detect-key.pem

# Connect to the remote instance
ssh -i "fraud-detect-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

## Step 2: Install Container Runtime Environment

Once authenticated within the remote Ubuntu shell, initialize and configure the Docker engine:

```bash
# Update local package indexes
sudo apt-get update

# Install the standard Docker runtime
sudo apt-get install -y docker.io

# Enable the Docker daemon to automatically initialize on system boot
sudo systemctl start docker
sudo systemctl enable docker

# Add the default ubuntu user to the docker group to execute commands without sudo
sudo usermod -aG docker $USER

# CRITICAL: Terminate session and reconnect via SSH for group updates to take effect
exit
```

## Step 3: Deploy the Fraud Detection Service

Reconnect to your EC2 instance and run the following deployment script to build the image layer and run the container with persistence guardrails:

```bash
# 1. Clone the production source code from the repository
git clone https://github.com/RedSamurai07/Fraud-Detection-using-Anomaly-Detection-project.git
cd Fraud-Detection-using-Anomaly-Detection-project

# 2. Build the Docker application image layer
docker build -t fraud-detection-api .

# 3. Instantiate the production container engine
# Maps runtime ports, ensures data persistence, and establishes crash auto-restart logic
docker run -d \
  -p 5000:5000 \
  -p 7860:7860 \
  -v mlflow_data:/app/mlruns \
  -v model_artifacts:/app/models \
  --name fraud-service \
  --restart unless-stopped \
  fraud-detection-api
```

💡 **Production Enhancements Added:**

* `--restart unless-stopped`: Ensures the fraud detection service automatically reboots if the application crashes or the underlying EC2 server undergoes a hardware reboot.

* `-v mlflow_data:/app/mlruns`: Mounts a persistent named Docker volume so your tracked MLflow experiment metadata and evaluation metrics survive container updates and deletions.

* `-v model_artifacts:/app/models`: Preserves trained model artifacts (`best_model.joblib`, `scaler.joblib`) across container lifecycle events, ensuring inference continuity.

## Step 4: Infrastructure & Service Verification

**1. Health-Check Endpoint API Validation**

Test the baseline responsiveness of the FastAPI engine from your local machine terminal:

```bash
curl http://<EC2_PUBLIC_IP>:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

**2. Fraud Prediction Endpoint Validation**

Submit a test transaction payload to verify the inference pipeline is operational:

```bash
curl -X POST http://<EC2_PUBLIC_IP>:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
    "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
    "V9": 0.3637, "V10": 0.0907, "V11": -0.5515, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
    "Amount": 149.62
  }'
```

**3. Interactive Swagger API Playground**

FastAPI natively serves interactive OpenAPI documentation. You can test live fraud prediction payloads directly through your web browser at:

```
http://<EC2_PUBLIC_IP>:5000/docs
```

**4. AI-Powered Gradio Detection Interface**

To interact with the visual fraud detection interface featuring transaction input fields, sample data, and real-time analysis results, navigate to:

```
http://<EC2_PUBLIC_IP>:7860
```

## CI/CD Pipeline Status

The operational integrity of the master codebase is continuously protected via automated integration testing gates:

![CI Pipeline](https://github.com/RedSamurai07/Fraud-Detection-using-Anomaly-Detection-project/actions/workflows/main.yml/badge.svg)
