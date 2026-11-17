# 📊 Employee Attrition Prediction Using Data Mining Techniques (DMBI CA2)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/DMBI_CA2-Complete-brightgreen)

**Author:** Prithviraj Virwani  
**Class:** D15B | **Roll No:** 70  
**Course:** Data Mining & Business Intelligence (DMBI) — CA2 Mini Project (10 Marks)

---

## 🎯 Problem Statement & Business Goal
Employee attrition costs enterprises **1.5x to 2x annual salary** per departing employee. This project implements a complete Data Mining & Business Intelligence pipeline to:
1. Predict flight-risk employees before resignation using supervised classification models.
2. Identify key attrition drivers (OverTime, MonthlyIncome, StockOptionLevel, Early Tenure).
3. Translate predictions into proactive HR retention strategies with quantifiable ROI.

---

## 📂 Repository Contents
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — IBM HR Analytics Employee Attrition Dataset (1,470 rows, 35 columns)
- `DMBI_CA2_Employee_Attrition.ipynb` — Interactive Jupyter / Google Colab Notebook
- `employee_attrition_mining.py` — Standalone Python Data Mining & Visualization script
- `DMBI_CA2_REPORT_PRITHVIRAJ_VIRWANI.html` — Printable PDF-ready Academic Report with 6 visualizations
- `DMBI_CA2_REPORT_PRITHVIRAJ_VIRWANI.md` — Markdown Report
- `requirements.txt` — Python dependencies

---

## 📈 Model Evaluation Results (Stratified 80/20 Test Set, N = 294)

| Data Mining Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree (CART, depth=5) | 80.61% | 43.24% | 68.09% | 52.89% | 0.77 |
| **Random Forest (200 Trees)** | **87.76%** | **60.00%** | 70.21% | **64.71%** | **0.84** |
| Logistic Regression (Scaled) | 84.69% | 51.47% | **74.47%** | 60.87% | 0.83 |

---

## 🔍 Top Business Insights
1. **OverTime = 2.92x Attrition Risk:** 30.53% attrition among overtime staff vs 10.44% non-overtime.
2. **Compensation Gap:** Departing staff average **$4,787/month** vs **$6,833/month** for retained staff (~30% lower).
3. **Stock Options Anchor Loyalty:** Level 0 stock holders leave at **24.41%** vs **7.59%** for Level 2 holders.
4. **Department Hotspots:** Sales (**20.63%**) and HR (**19.05%**) lead turnover vs R&D (**13.84%**).

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python employee_attrition_mining.py
```
Or open `DMBI_CA2_Employee_Attrition.ipynb` directly in **Google Colab**.
