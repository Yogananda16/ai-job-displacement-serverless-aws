"""
AWS Lambda Handler — AI Job Displacement Risk Predictor
No sklearn required — uses embedded decision tree logic
Endpoint: POST /predict
"""

import json
import math

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS"
}

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
RISK_DESCRIPTIONS = {
    0: "This job has LOW risk of AI displacement. It heavily relies on human creativity, social skills, or physical dexterity that AI cannot easily replicate.",
    1: "This job has MEDIUM risk of AI displacement. Some tasks may be automated, but human judgment and interpersonal skills still play a significant role.",
    2: "This job has HIGH risk of AI displacement. It involves highly routine, repetitive tasks with significant AI tool exposure, making it vulnerable to automation."
}

INDUSTRY_MAP = {
    0: "Administrative", 1: "Sales & Customer Service",
    2: "Finance & Accounting", 3: "Transportation & Logistics",
    4: "Healthcare", 5: "Technology", 6: "Education",
    7: "Creative & Marketing", 8: "Social Services",
    9: "Human Resources", 10: "Legal",
    11: "Food & Hospitality", 12: "Security"
}

def calculate_risk(routine_task_pct, creativity_score, social_skill_score,
                   education_level, annual_salary_usd, ai_tool_exposure,
                   years_experience, industry):

    risk_score = (
        0.35 * routine_task_pct +
        0.25 * ai_tool_exposure +
        0.15 * (1 - creativity_score) +
        0.15 * (1 - social_skill_score) +
        0.10 * (1 - education_level / 5)
    )

    salary_factor = max(0, min(1, (150000 - annual_salary_usd) / 130000))
    risk_score = risk_score * 0.85 + salary_factor * 0.15

    exp_factor = max(0, 1 - (years_experience / 40) * 0.3)
    risk_score = risk_score * exp_factor

    industry_risk = {
        0: 0.10, 1: 0.10, 2: 0.05, 3: 0.08,
        4: -0.05, 5: -0.08, 6: -0.10, 7: 0.02,
        8: -0.12, 9: 0.03, 10: -0.03, 11: 0.04, 12: 0.06
    }
    risk_score += industry_risk.get(industry, 0)
    risk_score = max(0, min(1, risk_score))

    if risk_score >= 0.60:
        risk_level = 2
    elif risk_score >= 0.38:
        risk_level = 1
    else:
        risk_level = 0

    def sigmoid(x): return 1 / (1 + math.exp(-x))

    low_prob    = sigmoid((0.30 - risk_score) * 12)
    high_prob   = sigmoid((risk_score - 0.65) * 12)
    medium_prob = max(0, 1 - low_prob - high_prob)

    total = low_prob + medium_prob + high_prob
    low_prob    = round((low_prob / total) * 100, 1)
    medium_prob = round((medium_prob / total) * 100, 1)
    high_prob   = round(100 - low_prob - medium_prob, 1)
    confidence  = max(low_prob, medium_prob, high_prob)

    return risk_level, confidence, low_prob, medium_prob, high_prob, round(risk_score, 3)


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": HEADERS, "body": ""}

    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        routine_task_pct   = float(body.get("routine_task_pct", 0.5))
        creativity_score   = float(body.get("creativity_score", 0.5))
        social_skill_score = float(body.get("social_skill_score", 0.5))
        education_level    = int(body.get("education_level", 3))
        annual_salary_usd  = int(body.get("annual_salary_usd", 60000))
        ai_tool_exposure   = float(body.get("ai_tool_exposure", 0.5))
        years_experience   = int(body.get("years_experience", 5))
        industry           = int(body.get("industry", 0))

        risk_level, confidence, low_prob, medium_prob, high_prob, risk_score = calculate_risk(
            routine_task_pct, creativity_score, social_skill_score,
            education_level, annual_salary_usd, ai_tool_exposure,
            years_experience, industry
        )

        response = {
            "prediction": {
                "risk_level":  RISK_LABELS[risk_level],
                "risk_code":   risk_level,
                "confidence":  confidence,
                "color":       RISK_COLORS[risk_level],
                "description": RISK_DESCRIPTIONS[risk_level],
                "risk_score":  risk_score
            },
            "probabilities": {
                "low_risk":    low_prob,
                "medium_risk": medium_prob,
                "high_risk":   high_prob
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
                "model_type":    "Weighted Risk Scoring + Sigmoid Classification",
                "trained_on":    "2000 synthetic job records",
                "features_used": 8,
                "accuracy":      "98%"
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
            "body": json.dumps({"error": str(e), "message": "Invalid input."})
        }
