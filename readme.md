# 🛡️ Phishing Email Detection System

A comprehensive email phishing detection system that uses multiple AI/ML models to classify emails, attachments, and links for potential phishing threats.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ML_Framework-red.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Setup Instructions](#setup-instructions)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. API Keys Configuration](#3-api-keys-configuration)
  - [4. Running the Application](#4-running-the-application)
- [Usage](#usage)
- [Model Details](#model-details)

---

## Overview

This is an intelligent email security system that analyzes emails from Gmail and Outlook for phishing threats using a multi-layered approach:

- **Email Body Classification** - RoBERTa-based deep learning model
- **Document Attachment Classification** - RoBERTa + LoRA model for sensitive content detection
- **Image Attachment Classification** - CNN-based image classification
- **Link Analysis** - Domain reputation checking against trusted domains database
- **Malware Detection** - YARA rule-based scanning for malicious patterns

---

## Features

✅ **Gmail & Outlook Integration** - Connect your email accounts via OAuth  
✅ **Real-time Email Classification** - Automatic phishing detection  
✅ **Multi-format Document Analysis** - PDF, DOCX, CSV, Excel, TXT support  
✅ **Image Scanning** - Detect sensitive or malicious images  
✅ **YARA Malware Scanning** - Detect known malware signatures  
✅ **URL Reputation Check** - Verify links against 1M+ trusted domains  
✅ **User Feedback System** - Improve accuracy with user corrections  

---

## Project Architecture

```
PHISHING DETECTION MAIN/
├── app.py                          # Main Flask application
├── body_classifier.py              # RoBERTa email body classifier
├── document_classifier.py          # RoBERTa document attachment classifier
├── requirements.txt                # Python dependencies
├── top-1m.csv                      # Trusted domains database
├── image_model.h5                  # CNN image classification model
├── roberta_lora_phishing_detector.pt   # Email body RoBERTa model
├── Data Classification File and Model/
│   └── best_roberta_model_2.2M_1_Epoc.pt  # Document classifier model
├── awesome-yara/
│   └── rules/                      # YARA malware detection rules
└── templates/                      # HTML templates
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### 2. Download Required Files

> ⚠️ **Important:** The ML models and YARA rules are not included in this repository due to their large size. You must download them separately.

📥 **[Download Required Project Files from Google Drive](https://drive.google.com/file/d/1mFBAFdXUWLhbsOluyEVSzfkO58_Gg9HF/view?usp=sharing)**

After downloading, extract the files and place them in the project root directory. The download includes:
- `image_model.h5` - CNN image classification model
- `roberta_lora_phishing_detector.pt` - Email body RoBERTa model
- `Data Classification File and Model/` - Document classifier model
- `awesome-yara/` - YARA malware detection rules

### 3. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/AyushGupta1332/ANU_PARAM_SHIELD.git
cd "PHISHING DETECTION MAIN"

# Install Python packages
pip install -r requirements.txt

# Download NLTK data (required for sentence tokenization)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. API Keys Configuration

You need to set up API keys for Gmail and Outlook integration. Follow these steps:

#### 📧 Google Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services** → **Library**
4. Search for "Gmail API" and enable it
5. Go to **APIs & Services** → **Credentials**
6. Click **Create Credentials** → **OAuth 2.0 Client IDs**
7. Set Application type to **Web application**
8. Add `http://127.0.0.1:5000/callback` to **Authorized redirect URIs**
9. Copy your **Client ID** and **Client Secret**

#### 📧 Microsoft Outlook API Setup

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Click **New registration**
4. Set redirect URI to `http://localhost:5000/callback_outlook` (Web type)
5. Under **Certificates & secrets**, create a new client secret
6. Under **API permissions**, add `Microsoft Graph` → `Mail.Read`
7. Copy your **Application (client) ID** and **Client Secret**

#### 🔧 Setting Environment Variables

Open **PowerShell** (as Administrator if needed) and run:

```powershell
# Google Gmail API
setx GOOGLE_CLIENT_ID "your-google-client-id"
setx GOOGLE_CLIENT_SECRET "your-google-client-secret"
setx GOOGLE_REDIRECT_URI "http://127.0.0.1:5000/callback"

# Microsoft Outlook API
setx OUTLOOK_CLIENT_ID "your-outlook-client-id"
setx OUTLOOK_CLIENT_SECRET "your-outlook-client-secret"
setx OUTLOOK_REDIRECT_URI "http://localhost:5000/callback_outlook"
```

> ⚠️ **Important:** After running `setx` commands, close and reopen your terminal/VS Code for changes to take effect.

### 4. Running the Application

```bash
# Navigate to project directory
cd "PHISHING DETECTION MAIN"

# Run the Flask application
python app.py
```

The application will start at `http://127.0.0.1:5000`

---

## Usage

1. Open `http://127.0.0.1:5000` in your browser
2. Choose to connect **Gmail** or **Outlook**
3. Authorize access to your email account
4. View your emails classified as:
   - ✅ **Safe** - Legitimate emails
   - ⚠️ **Phishing** - Potential phishing threats
   - 🔍 **Needs Review** - Uncertain, requires manual review
5. Click on any email to see detailed analysis
6. Download and scan attachments for malware
7. Provide feedback to improve classification accuracy

---

## Model Details

| Component | Model | Purpose |
|-----------|-------|---------|
| Email Body | RoBERTa + LoRA | Classifies email text as phishing/safe |
| Document Attachments | RoBERTa + LoRA | Classifies documents as sensitive/non-sensitive |
| Image Attachments | CNN (Keras) | Classifies images as sensitive/non-sensitive |
| Link Analysis | Rule-based | Checks URLs against trusted domains |
| Malware Detection | YARA Rules | Scans for known malware patterns |

---

## License

This project is developed for educational purposes as part of a Year Project.

---

## Contributors

- Ayush Gupta

---

**Made with ❤️ for email security**
