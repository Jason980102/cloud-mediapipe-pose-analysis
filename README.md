# Cloud-Based Intelligent Taekwondo Motion Classification System

## Overview

This project demonstrates a cloud-based intelligent taekwondo motion classification system using MediaPipe Pose estimation, machine learning, and multi-cloud deployment architecture.

The system supports:
- Video upload
- MediaPipe pose extraction
- AI-based taekwondo motion classification
- Streamlit frontend interaction
- AWS S3 cloud storage
- Oracle Cloud backup integration

---

## System Architecture

### Google Cloud / Google Colab
- Dataset preprocessing
- MediaPipe keypoint extraction
- Human labeling workflow
- Feature engineering
- AI model training

### AWS EC2 + Streamlit
- Real-time frontend service
- Video upload interface
- AI inference pipeline
- MediaPipe pose preview generation

### AWS S3
- Model artifact storage
- Prediction result storage
- MediaPipe preview storage

### Oracle Cloud Object Storage
- Cross-cloud backup node
- Secondary storage layer

---

## Technologies

- Python
- Streamlit
- MediaPipe
- OpenCV
- Scikit-learn
- AWS EC2
- AWS S3
- Oracle Cloud Object Storage
- Google Colab

---

## Repository Structure

```text
notebooks/          # Colab preprocessing and training notebooks
src/                # Pose processing source code
taekwondo-demo/     # Streamlit frontend application
