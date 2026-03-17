"""
AI Job Displacement Risk Predictor
Dataset: Synthetic, based on real research from Oxford, McKinsey, WEF reports
Model: Random Forest Classifier
Output: model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 2000  # number of synthetic job records

# ── Feature definitions ───────────────────────────────────────────────────────
# Based on Oxford Future of Employment study (Frey & Osborne) + McKinsey 2024

job_profiles = {
    # (routine_task_pct, creativity_score, social_skill, education_req, salary_usd, ai_tool_exposure, industry_code)
    "Data Entry Clerk":         (0.92, 0.10, 0.15, 1, 35000,  0.95, 0),
    "Telemarketer":             (0.88, 0.12, 0.40, 1, 32000,  0.90, 1),
    "Accountant":               (0.65, 0.35, 0.40, 3, 72000,  0.75, 2),
    "Truck Driver":             (0.70, 0.15, 0.20, 1, 48000,  0.80, 3),
    "Radiologist":              (0.50, 0.55, 0.55, 5, 320000, 0.70, 4),
    "Software Engineer":        (0.30, 0.80, 0.55, 4, 120000, 0.60, 5),
    "Nurse":                    (0.35, 0.45, 0.90, 3, 75000,  0.30, 4),
    "Teacher":                  (0.25, 0.70, 0.95, 4, 58000,  0.25, 6),
    "Graphic Designer":         (0.40, 0.90, 0.50, 3, 55000,  0.65, 7),
    "Warehouse Worker":         (0.80, 0.10, 0.15, 1, 38000,  0.85, 3),
    "Financial Analyst":        (0.60, 0.55, 0.50, 4, 95000,  0.70, 2),
    "Social Worker":            (0.20, 0.50, 0.95, 4, 50000,  0.15, 8),
    "HR Manager":               (0.45, 0.55, 0.80, 4, 78000,  0.50, 9),
    "Customer Service Rep":     (0.82, 0.15, 0.60, 2, 36000,  0.88, 1),
    "Lawyer":                   (0.35, 0.80, 0.75, 5, 130000, 0.55, 10),
    "Cook / Chef":              (0.55, 0.65, 0.45, 1, 34000,  0.40, 11),
    "Pharmacist":               (0.60, 0.45, 0.65, 5, 128000, 0.60, 4),
    "Marketing Manager":        (0.30, 0.80, 0.80, 4, 92000,  0.50, 7),
    "Security Guard":           (0.75, 0.10, 0.35, 1, 31000,  0.70, 12),
    "Research Scientist":       (0.20, 0.95, 0.60, 5, 115000, 0.40, 5),
}

# ── Generate synthetic rows ───────────────────────────────────────────────────
rows = []
job_names = list(job_profiles.keys())

for _ in range(N):
    job = np.random.choice(job_names)
    rt, cr, ss, ed, sal, ai_exp, ind = job_profiles[job]

    # Add realistic noise
    routine_task_pct   = np.clip(np.random.normal(rt,   0.06), 0, 1)
    creativity_score   = np.clip(np.random.normal(cr,   0.06), 0, 1)
    social_skill_score = np.clip(np.random.normal(ss,   0.06), 0, 1)
    education_level    = int(np.clip(np.random.normal(ed, 0.5), 1, 5))
    annual_salary_usd  = int(np.clip(np.random.normal(sal, sal * 0.10), 20000, 400000))
    ai_tool_exposure   = np.clip(np.random.normal(ai_exp, 0.05), 0, 1)
    years_experience   = int(np.clip(np.random.normal(8, 4), 0, 40))
    industry           = ind

    # ── Risk label (based on research-backed formula) ─────────────────────────
    risk_score = (
        0.35 * routine_task_pct +
        0.25 * ai_tool_exposure +
        0.15 * (1 - creativity_score) +
        0.15 * (1 - social_skill_score) +
        0.10 * (1 - education_level / 5)
    )

    if risk_score >= 0.60:
        risk_level = 2   # High
    elif risk_score >= 0.38:
        risk_level = 1   # Medium
    else:
        risk_level = 0   # Low

    rows.append({
        "job_title":          job,
        "routine_task_pct":   round(routine_task_pct, 3),
        "creativity_score":   round(creativity_score, 3),
        "social_skill_score": round(social_skill_score, 3),
        "education_level":    education_level,
        "annual_salary_usd":  annual_salary_usd,
        "ai_tool_exposure":   round(ai_tool_exposure, 3),
        "years_experience":   years_experience,
        "industry":           industry,
        "risk_level":         risk_level,
    })

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows")
print(df["risk_level"].value_counts().rename({0:"Low",1:"Medium",2:"High"}))

# ── Train model ───────────────────────────────────────────────────────────────
FEATURES = [
    "routine_task_pct", "creativity_score", "social_skill_score",
    "education_level", "annual_salary_usd", "ai_tool_exposure",
    "years_experience", "industry"
]

X = df[FEATURES]
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n📊 Model Performance:")
print(classification_report(y_test, model.predict(X_test),
      target_names=["Low Risk", "Medium Risk", "High Risk"]))

# ── Save model ────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)

print("✅ model.pkl saved!")
