# 🔐 Password Strength Detection — ML Edition

> *"Your password's biggest enemy isn't a hacker — it's `password123`."*

**Author:** Zaid Bharde

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

A machine learning system that looks at a password the way an attacker would — length, entropy, patterns, common-password overlap — and tells you instantly if it's **Weak 🔴 / Medium 🟡 / Strong 🟢**.

Started as a simple TF-IDF classifier. Evolved into a full security-feature pipeline with entropy scoring, pattern detection, and common-password intelligence baked in.

---

## ⚡ What it does

Type a password → get an instant strength verdict, backed by real security heuristics, not just a black-box guess.

## 🗂️ Project Structure

```text
.
├── data/
│   ├── Password Strength.csv
│   └── common_passwords.txt
├── models/
│   └── password_strength_xgboost.joblib
├── notebooks/
│   └── Checking Password Strength.ipynb
├── feature_engineering.py
├── train_xgboost.py
├── password_strength.py
├── password_strength_gui.py
└── README.md
```

## 📊 Dataset

- **Password** – 670k+ unique passwords collected online
- **Strength** – 0 (weak), 1 (medium), 2 (strong)

## 🧠 Feature Engineering

Every password gets fingerprinted using:

| Category | Features |
|---|---|
| **Text patterns** | Character-level TF-IDF n-grams |
| **Basics** | Length, Shannon entropy score |
| **Character mix** | Upper/lower/digit/special flags + counts, diversity score |
| **Attacker tricks** | Sequential patterns (`abc`, `123`, qwerty rows), repeated chars (`aaa`, `111`) |
| **Real-world risk** | Common password exact-match, dictionary word containment |

## 🏆 Models & Performance

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~81.8% |
| Gradient Boosting | ~91.3% |
| **XGBoost + TF-IDF + engineered features** | **100.0%** 🚩 |

```text
Rows: 669,639
TF-IDF features: 2,500
Engineered features: 16
Total features: 2,516
XGBoost — Train Acc: 1.0000, Test Acc: 1.0000
```

### 🚩 Real Talk — Why 100% Isn't a Victory Lap

A model that's *never* wrong is usually a model that's *cheating*. This dataset's labels look rule-derived from the same properties (length, diversity, common-password membership) that are now feeding the model as features — so XGBoost isn't learning security, it's learning the labeling rule.

**Before this goes on a resume as "100% accurate," here's the validation debt:**
- [ ] Test on an independently labeled, real-world password dataset
- [ ] Ablation study — TF-IDF only vs. engineered-only vs. combined — to isolate the leak
- [ ] Manual spot-check of predictions against the original rule logic

## 🚀 Usage

**Desktop GUI:
```bash
python password_strength_gui.py
```
Trains Logistic Regression on launch, then checks strength live as you type.

The GUI loads the saved XGBoost model from `models/` when available and falls back to Logistic Regression training if the artifact is missing. It includes a modern dashboard, password generator, animated score meter, entropy/crack-time estimates, character breakdown, and actionable feedback.

**CLI:**
```bash
python password_strength.py
```
or a one-shot check:
```bash
python password_strength.py --password "ExamplePass123!"
```

Switch to Gradient Boosting:
```bash
python password_strength.py --model gradient-boosting
```

**Train the XGBoost model:**
```bash
python train_xgboost.py
```

## 📦 Dependencies

`pandas` `numpy` `seaborn` `scikit-learn` `scipy` `xgboost` `joblib`

## 🛣️ Roadmap

- [x] TF-IDF baseline (Logistic Regression, Gradient Boosting)
- [x] Security-focused feature engineering
- [x] XGBoost model on combined features
- [ ] Validate on independent real-world labeled data
- [ ] SHAP explainability — *why* is this password weak?
- [ ] FastAPI backend (`/check-password`)
- [ ] Have I Been Pwned breach check (k-anonymity, hash-prefix only — never sends raw password)
- [ ] Live web frontend with strength meter + breach alert

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE)

---

<p align="center"><i>Built to make "123456" a little more embarrassing.</i></p>
