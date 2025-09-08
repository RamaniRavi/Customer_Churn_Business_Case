# Customer Churn — Business Case (Telecom)

Machine learning project to predict and explain telecom customer churn for **No-Churn Telecom**.  
The model generates a **CHURN_FLAG** (YES=1 / NO=0) and a calibrated **risk score** to target at-risk customers with retention offers, prioritize support tickets, and guide pricing & service decisions.

> Current churn rate: **14%** (656 of 4,617 customers).  
This repository helps identify *who* is likely to churn and *why*, so the business can act before customers leave.

---

## 🎯 Project Goals

1. **Identify drivers of churn** — understand which variables most influence churn.  
2. **Predict churn risk** — produce a binary **CHURN_FLAG** and probability **risk score (0–1)**.  
3. **Enable retention actions** — email campaigns, CRM flags, high-priority support tickets.  

---

## 📂 Repository Structure

```
customer-churn-business-case/
│
├── data/
│   └── PRCL-0017_Customer_Churn_Business_case_dataset.csv
│
├── notebooks/
│   └── PRCL-0017_Customer_Churn_Business_case.ipynb
│
├── models/
│   └── PRCL-0017_Churn.pkl
│
├── reports/
│   ├── business/
│   │   ├── Customer Churn Report.pdf
│   │   ├── Customer Churn Report.docx
│   │   └── PRCL-0017 Customer Churn Business case (1).pdf
│   ├── dashboards/
│   │   └── customer churn dashboard.pdf
│   └── visuals/   # plots (feature importance, confusion matrix, etc.)
│
├── requirements.txt
└── README.md
```

---

## 🧾 Data Dictionary (key features)

- **Account/Plan info**: `State`, `Account Length`, `Area Code`, `International Plan`, `VMail Plan`, `VMail Message`  
- **Usage metrics**: `Day/Eve/Night/International Mins`, `Calls`, `Charges`  
- **Support interaction**: `CustServ_Calls`  
- **Target**: `Churn` (historical)  

> Charges are linear transforms of minutes; retain only one to avoid multicollinearity.

---

## 🔑 Insights (EDA + Dashboard)

- **Voice Mail Plan**: Customers without it = **85% of churn**.  
- **International Plan**: 72% of churned customers lack this plan.  
- **High usage customers** (Day/Eve minutes & charges) → higher churn risk.  
- **Customer Service Calls**: Strong churn driver — dissatisfaction indicator.  
- **Geography**: States like NJ, TX, WA, WV, and MD have highest churn.  

**Recommendations**:  
- Offer/promote **Voice Mail** & **International Plans**.  
- Target **high-churn states** with localized campaigns.  
- Improve **customer service resolution** to reduce repeat calls.  
- Adjust pricing for **heavy users** with bundles/loyalty programs.  

---

## 📊 Model Evaluation Summary

### Before Hyperparameter Tuning
- **Random Forest / Gradient Boosting / XGBoost / LightGBM**: ~91% accuracy, F1 up to 67%.  
- **Bagging**: Balanced performance, best F1 (67.2%).  
- **SVM & KNN**: Poor test performance.  

### After Hyperparameter Tuning
- **LightGBM**: Best — **94.1% accuracy**, **F1 = 78.2%**, strong generalization.  
- **Gradient Boosting**: Also strong — 93.5% accuracy, F1 = 75.2%.  
- **Bagging**: Balanced, but lower (87.5% accuracy, F1 = 60.5%).  
- **Logistic Regression, SVM, KNN**: Underperformed.  

**Deployment recommendation**:  
Use **LightGBM** as primary model, with Bagging/Gradient Boosting as fallback options.  

---

## ⚙️ Setup

```bash
# create venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**requirements.txt (core):**
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
```

Optional: `shap`, `pdpbox` for explainability.

---

## ▶️ Quickstart

### Train & Evaluate
Open `notebooks/PRCL-0017_Customer_Churn_Business_case.ipynb` and run cells:  
- Ingestion → EDA → Preprocessing → Modeling → Evaluation → Export.  

### Batch Inference
```python
import pickle, pandas as pd

model = pickle.load(open("models/PRCL-0017_Churn.pkl","rb"))
X = pd.read_csv("data/raw/new_customers.csv")

proba = model.predict_proba(X)[:,1]
X["churn_risk"] = proba
X["CHURN_FLAG"] = (proba >= 0.35).astype(int)

X.to_csv("scored_customers.csv", index=False)
```

---

## 📈 Business Activation

- **Marketing**: retention offers to top-N% risk customers.  
- **Customer Care**: auto-prioritize tickets from high-risk accounts.  
- **Regional targeting**: focus on high-churn states.  
- **Pricing**: bundles/loyalty discounts for heavy users.  

---

## 🧩 Governance

- **Monitor drift** (features & target).  
- **Retrain regularly** to prevent model decay.  
- **Fairness checks** across states & plans.  
- **GDPR compliance**: exclude/secure PII (phone/account IDs).  

---

## Attribution
- Project built using anonymized No-Churn Telecom dataset and business context.
