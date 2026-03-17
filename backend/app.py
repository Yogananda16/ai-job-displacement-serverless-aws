"""
AWS Lambda Handler — AI Job Displacement Risk Predictor
Endpoint: POST /predict
Input:  JSON with job features
Output: JSON with risk level + confidence + explanation
"""

import json
import pickle
import os
import boto3

# ── Load model once (outside handler = cached across Lambda invocations) ──────
MODEL_PATH = "/tmp/model.pkl" if os.path.exists("/tmp/model.pkl") else "model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]

model, FEATURES = load_model()

# ── Risk label map ─────────────────────────────────────────────────────────────
RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

RISK_COLORS = {
    0: "#22c55e",   # green
    1: "#f59e0b",   # amber
    2: "#ef4444"    # red
}

RISK_DESCRIPTIONS = {
    0: "This job has low risk of AI displacement. It heavily relies on human creativity, social skills, or physical dexterity that AI cannot easily replicate.",
    1: "This job has moderate risk of AI displacement. Some tasks may be automated, but human judgment and interpersonal skills still play a significant role.",
    2: "This job has high risk of AI displacement. It involves highly routine, repetitive tasks with significant AI tool exposure, making it vulnerable to automation."
}

# ── Industry name map ──────────────────────────────────────────────────────────
INDUSTRY_MAP = {
    0: "Administrative",
    1: "Sales & Customer Service",
    2: "Finance & Accounting",
    3: "Transportation & Logistics",
    4: "Healthcare",
    5: "Technology",
    6: "Education",
    7: "Creative & Marketing",
    8: "Social Services",
    9: "Human Resources",
    10: "Legal",
    11: "Food & Hospitality",
    12: "Security"
}

# ── CORS headers ───────────────────────────────────────────────────────────────
HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS"
}

# ── Main Lambda handler ────────────────────────────────────────────────────────
def lambda_handler(event, context):
    # Handle CORS preflight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 200, "headers": HEADERS, "body": ""}

    try:
        # Parse request body
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        # Extract and validate features
        routine_task_pct   = float(body.get("routine_task_pct", 0.5))
        creativity_score   = float(body.get("creativity_score", 0.5))
        social_skill_score = float(body.get("social_skill_score", 0.5))
        education_level    = int(body.get("education_level", 3))
        annual_salary_usd  = int(body.get("annual_salary_usd", 60000))
        ai_tool_exposure   = float(body.get("ai_tool_exposure", 0.5))
        years_experience   = int(body.get("years_experience", 5))
        industry           = int(body.get("industry", 0))

        # Build feature vector
        features = [[
            routine_task_pct,
            creativity_score,
            social_skill_score,
            education_level,
            annual_salary_usd,
            ai_tool_exposure,
            years_experience,
            industry
        ]]

        # Predict
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        confidence = round(max(probabilities) * 100, 1)

        # Build response
        response = {
            "prediction": {
                "risk_level": RISK_LABELS[prediction],
                "risk_code": prediction,
                "confidence": confidence,
                "color": RISK_COLORS[prediction],
                "description": RISK_DESCRIPTIONS[prediction]
            },
            "probabilities": {
                "low_risk":    round(probabilities[0] * 100, 1),
                "medium_risk": round(probabilities[1] * 100, 1),
                "high_risk":   round(probabilities[2] * 100, 1)
            },
            "input_summary": {
                "routine_task_pct":   routine_task_pct,
                "creativity_score":   creativity_score,
                "social_skill_score": social_skill_score,
                "education_level":    education_level,
                "annual_salary_usd":  annual_salary_usd,
                "ai_tool_exposure":   ai_tool_exposure,
                "years_experience":   years_experience,
                "industry":           INDUSTRY_MAP.get(industry, "Unknown")
            },
            "model_info": {
                "model_type": "Random Forest Classifier",
                "trained_on": "2000 synthetic job records",
                "features_used": len(FEATURES),
                "accuracy": "98%"
            }
        }

        return {
            "statusCode": 200,
            "headers": HEADERS,
            "body": json.dumps(response)
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({
                "error": str(e),
                "message": "Invalid input. Please check your request body."
            })
        }
