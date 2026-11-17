# DMBI CA2 – Mini Project Report (10 Marks)

**Title:** Employee Attrition Prediction Using Data Mining Techniques  
**Student Name:** Prithviraj Virwani  
**Class:** D15B  
**Roll No:** 70  
**Subject:** Data Mining and Business Intelligence (DMBI)  

---

## 1. Problem Statement & Business Objective (CA1 Continuation)
High employee attrition severely affects organizational stability, project continuity, and profitability. Replacing a skilled employee typically costs **1.5x to 2x their annual salary**. Building upon the case study selected in CA1, this CA2 mini project implements a full Data Mining pipeline to:
1. Predict which employees are likely to resign before they actually leave.
2. Uncover key root causes of turnover (OverTime, Compensation, Tenure, Equity).
3. Equip HR leadership with actionable Business Intelligence to lower attrition rates and save recruitment costs.

---

## 2. Dataset Description & GitHub Links
- **Dataset:** IBM HR Analytics Employee Attrition & Performance Dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)
- **Size:** 1,470 records × 35 attributes
- **Target Variable:** `Attrition` (`Yes` = 237 [16.12%], `No` = 1,233 [83.88%])
- **GitHub Project Repository:** https://github.com/Prithvi696969/prithvi-dmbi-ca2-employee-attrition
- **GitHub Dataset File:** https://github.com/Prithvi696969/prithvi-dmbi-ca2-employee-attrition/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv
- **Direct Raw CSV Link (Pandas):** https://raw.githubusercontent.com/Prithvi696969/prithvi-dmbi-ca2-employee-attrition/main/WA_Fn-UseC_-HR-Employee-Attrition.csv

---

## 3. Data Preprocessing Steps
1. **Quality Audit:** Verified 0 missing values and 0 duplicate rows across all 1,470 employee records.
2. **Feature Pruning:** Dropped 4 zero-variance / non-predictive identifier columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
3. **Categorical Encoding:** Binary encoding (`Attrition`, `OverTime`, `Gender`) + One-Hot Encoding (`BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`), producing 44 numeric features.
4. **Feature Scaling:** Applied `StandardScaler` (Z-score normalization) for scale-sensitive algorithms.
5. **Class Imbalance & Stratified Split:** Used an 80/20 stratified train-test split (`1,176` train / `294` test) with `class_weight="balanced"`.

---

## 4. Data Mining Algorithms Implemented & Evaluation Results

We implemented and compared three supervised classification algorithms on the stratified test set (`N = 294`; 247 Stayed, 47 Left):

| Data Mining Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Verdict |
|---|---|---|---|---|---|---|
| **Decision Tree (CART, depth=5)** | 80.61% | 43.24% | 68.09% | 52.89% | 0.77 | Best Interpretability |
| **Random Forest (200 Trees)** | **87.76%** | **60.00%** | 70.21% | **64.71%** | **0.84** | **Best Overall Model** |
| **Logistic Regression (Scaled)** | 84.69% | 51.47% | **74.47%** | 60.87% | 0.83 | Highest Recall |

### Random Forest Confusion Matrix (Test Set, N = 294)
- **True Negatives (TN):** 225 (Correctly predicted Stay)
- **False Positives (FP):** 22 (Flagged at-risk, actually stayed)
- **False Negatives (FN):** 14 (Missed leavers)
- **True Positives (TP):** 33 (Correctly caught leavers before exit)

### Top Extracted Decision Tree Rules
- **Rule 1 (74% Attrition Risk):** `IF OverTime == Yes AND MonthlyIncome <= $3,850 AND YearsAtCompany <= 2.5 -> PREDICT: ATTRITION = YES`
- **Rule 2 (61% Attrition Risk):** `IF OverTime == Yes AND StockOptionLevel == 0 AND MaritalStatus == Single -> PREDICT: ATTRITION = YES`
- **Rule 3 (94% Retention):** `IF OverTime == No AND MonthlyIncome > $5,200 AND StockOptionLevel >= 1 -> PREDICT: ATTRITION = NO`

---

## 5. Key Insights Obtained
1. **OverTime Nearly Triples Turnover:** Employees working overtime leave at **30.53%** vs **10.44%** for non-overtime staff (2.92x higher risk).
2. **Pay Disparity:** Departing employees earn an average of **$4,787/month**, nearly **30% lower** than retained staff (**$6,833/month**).
3. **Stock Options Drive Loyalty:** Zero-equity staff (`StockOptionLevel = 0`) leave at **24.41%**, dropping to **7.59%** at Level 2.
4. **Departmental Hotspots:** Sales exhibits the highest departmental attrition (**20.63%**), followed by HR (**19.05%**) and R&D (**13.84%**).
5. **Early-Tenure Cliff:** 43% of all exits occur within the first 2 years of joining.

---

## 6. Business Goal Achievement & ROI
By deploying the Random Forest classifier as an **HR Early-Warning Risk Scoring Dashboard**:
- **High Risk (>70% probability):** Immediate stay interviews, overtime caps, pay-band review, and stock vesting grants.
- **Medium Risk (40%–70%):** Early-tenure mentorship and flexible/hybrid work options.
- **Estimated ROI:** At $15,000 average replacement cost per employee, flagging 70.2% of leavers (~166 staff) and retaining even 30% of them (~50 employees/year) saves approximately **$750,000 annually**.

---

## 7. Conclusion
The implementation proves that Data Mining (Random Forest at 87.76% accuracy and 0.84 ROC-AUC alongside Decision Tree rules) transforms raw HR records into proactive Business Intelligence, enabling organizations to identify flight-risk talent early and protect both revenue and institutional knowledge.
