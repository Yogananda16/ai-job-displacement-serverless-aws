# AI Job Displacement Risk Predictor
### Serverless ML System — AWS Lambda + API Gateway + S3

![AWS](https://img.shields.io/badge/AWS-Serverless-orange) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![ML](https://img.shields.io/badge/ML-Risk%20Prediction-green)

## Live Demo
- **Dashboard:** http://ai-job-dashboard-yogananda.s3-website-us-east-1.amazonaws.com
  
## Project Overview
A fully serverless AI system that predicts the risk of job displacement due to AI automation. Built for the Data Science Capstone — Spring 2026.

## Architecture
```
User → S3 Dashboard → API Gateway → Lambda → Prediction → Response
```

## Tech Stack
- **AWS Lambda** — Serverless Python inference
- **API Gateway** — HTTP POST /predict endpoint
- **Amazon S3** — Static dashboard hosting
- **Python 3.12** — Prediction logic (no EC2)

## ML Model
- **Features:** Routine Task %, Creativity, Social Skills, AI Exposure, Education, Salary, Experience, Industry
- **Output:** Low / Medium / High displacement risk + confidence %
- **Training Data:** 2,000 synthetic job records
- **Accuracy:** 98%

## Project Structure
```
├── backend/
│   ├── app.py              # Lambda handler
│   ├── train_model.py      # Model training script
│   ├── model.pkl           # Trained ML model
│   └── requirements.txt
├── frontend/
│   └── index.html          # S3 dashboard
└── README.md
```

## Author
Yogananda Manjunath | Data Science Capstone Course 2026
