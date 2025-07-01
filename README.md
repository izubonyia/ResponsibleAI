# 💼 Fairness-Aware Loan Approval Using Synthetic Data

This mini-project demonstrates how to simulate a simple machine learning pipeline for loan approval prediction using synthetic data, with a focus on **Responsible AI** concepts including **fairness auditing**, **bias mitigation**, and **model explainability**.

---

## 📊 Dataset

The dataset is synthetically generated with the following features:

- `income`: Simulated from a normal distribution (mean = 50,000, std = 15,000)
- `loan_amount`: Simulated from a normal distribution (mean = 15,000, std = 5,000)
- `gender`: Randomly assigned as 'Male' or 'Female' (used as a sensitive attribute)
- `approved`: Label assigned as 1 if `income > loan_amount`, else 0

---

## 🎯 Objective

Train a binary classifier to predict whether a loan should be approved, **detect bias** across gender groups, **mitigate bias**, and **explain model behavior** using SHAP.

---

## ⚙️ Technologies Used

- Python
- scikit-learn
- fairlearn
- shap
- pandas, numpy, matplotlib

---

## 🧪 Project Workflow

1. **Simulate Data**
2. **Define Target Variable**
3. **Add Sensitive Attribute (Gender)**
4. **Train Baseline Model**
5. **Evaluate Fairness**
6. **Mitigate Bias using ExponentiatedGradient**
7. **Explain Predictions with SHAP**

---

## 📈 Key Metrics

- **Accuracy**: Overall model performance
- **Demographic Parity Difference**: Measures fairness gap across gender
- **SHAP Summary Plot**: Visualizes feature contributions

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Notebook

Open and run the notebook:

📓 `Responsible_AI_Template_with_Fairness_Chart.ipynb`

You can also open it in [Google Colab](https://colab.research.google.com/) for easy access.

---

## 🧠 What You'll Learn

- How to simulate ML problems with synthetic data
- How to measure and mitigate bias using Fairlearn
- How to explain ML predictions using SHAP
- The trade-offs between accuracy and fairness

---

## 📂 File Structure

```
.
├── Responsible_AI_Template_with_Fairness_Chart.ipynb
├── requirements.txt
└── README.md
```

---

## 💬 Author

Made with ❤️ by [Your Name]

---
