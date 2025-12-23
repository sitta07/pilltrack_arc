# PillTrack: MLOps Producer Hub

![Status](https://img.shields.io/badge/Status-Beta-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![DataOps](https://img.shields.io/badge/DataOps-Enabled-darkslategray?style=for-the-badge)
![AWS S3](https://img.shields.io/badge/Storage-AWS%20S3-orange?style=for-the-badge&logo=amazon-s3&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-Passing-success?style=for-the-badge)

---

## Overview

**PillTrack Producer Hub** is a streamlined **MLOps production pipeline** for medicine pack identification.
The system manages the **end-to-end lifecycle** of dataset ingestion, AI-assisted auditing, feature extraction, artifact versioning, and synchronization with production environments via **AWS S3**.

This hub is designed for **AI Operators and MLOps Engineers** who need reliability, traceability, and rapid deployment for vision-based healthcare systems.

---

## Architecture & Core Modules

The system follows a **Modular "Src-Layout" Design** to ensure scalability, testability, and clean separation of concerns.

### Core Logic (`src/`)

- **`engine.py`**: The AI Brain.
  - **YOLOv8 Segmentation** for precise object localization.
  - **DINOv2** for state-of-the-art feature extraction.
  - Generates **4-directional rotation-invariant vectors** (0°, 90°, 180°, 270°).

- **`analytics.py`**: AI Auditor.
  - Performs **PCA (Principal Component Analysis)** for visualization.
  - Calculates **Euclidean Distances** to detect confusion risks and outliers.
  - Provides actionable suggestions (Low Data, High Spread).

- **`cloud_manager.py`**: Cloud Integration.
  - Handles secure artifact synchronization with **AWS S3**.
  - Includes robust error handling and connection status checks.

- **`db_manager.py`**: Data Registry.
  - Manages local vector databases (`.pkl`) and metadata generation (`.json`).

- **`utils.py`**: Helpers.
  - Centralized configuration loading and path management.

### Controller
- **`app.py`**: Streamlit Dashboard.
  - Serves as the UI/Controller, orchestrating the interaction between the user and the backend modules.

---

##  Key Features

- **Production Synchronization**: One-click Push/Pull of models and vector DBs to AWS S3.
- **AI-Powered Audit**: Automatically detects dataset health issues (Imbalance, Confusion Risk, High Variance).
- **Robust Testing**: Fully unit-tested (`pytest`) covering Cloud logic, AI Engine, and Data flow.
- **Automated Feature Extraction**: Auto-crop and rotate images to build robust embeddings.
- **Secure**: Environment variables management via `.env` to protect credentials.

---

## Project Structure

```bash
.
├── app.py                  # Main Streamlit UI Controller
├── config.yaml             # System Configuration
├── pytest.ini              # Testing Configuration
├── requirements.txt        # Dependencies
├── .env                    # Secrets (Not committed)
├── src/                    # Source Code (Core Logic)
│   ├── __init__.py
│   ├── analytics.py        # Math & Stats Logic
│   ├── cloud_manager.py    # AWS S3 Handler
│   ├── db_manager.py       # File & DB Handler
│   ├── engine.py           # YOLO + DINOv2 Engine
│   └── utils.py            # Config & Path Helpers
├── tests/                  # Unit Tests
│   ├── __init__.py
│   ├── test_cloud.py       # Cloud Mock Tests
│   └── test_engine.py      # AI Engine Mock Tests
├── database/               # Local Vector DB & Logs
└── models/                 # Pre-trained Model Weights (.pt)


```
## Getting Started
1️⃣ Prerequisites
Python 3.9 or higher

AWS Account with S3 Access

2️⃣ Installation

```bash
# Clone repository
git clone https://github.com/sitta07/PillTrack-Producer-Pipeline.git

# Install dependencies
pip install -r requirements.txt
```
3️⃣ Configuration
Create a .env file in the root directory:

```bash
S3_BUCKET_NAME=your-production-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your-default-region
```

4️⃣ Verification (Run Tests)
Ensure everything is working correctly before running the UI:

```bash
# Run all unit tests
python -m pytest
```

5️⃣ Execution
Start the MLOps Dashboard:

```bash
streamlit run app.py
```

## 👨‍💻 Author

**Sitta Boonkaew**  
AI Engineer Intern @ AI SmartTech  

---

## 📄 License

© 2025 AI SmartTech. All Rights Reserved.
