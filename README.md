# Password Strength Detection In ML

**Author:** Zaid Bharde

This project predicts password strength as weak, medium, or strong using machine learning. It started with character-level TF-IDF features and now includes security-focused engineered features such as entropy, character class diversity, repeated character detection, sequential pattern detection, and common password checks.

## Project Structure
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

## Dataset
- **Password** - 670k+ unique passwords collected online
- **Strength** - 0 (weak), 1 (medium), 2 (strong)

## Feature Engineering
Passwords are converted to numerical features using:

- Character-level TF-IDF n-grams
- Password length
- Shannon entropy score
- Uppercase, lowercase, digit, and special-character flags
- Uppercase, lowercase, digit, and special-character counts
- Character class diversity score
- Sequential pattern detection such as `abc`, `123`, and qwerty-row sequences
- Repeated character detection such as `aaa` and `111`
- Common password exact-match check
- Common dictionary word containment check

## Models & Performance
| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~81.8% |
| Gradient Boosting | ~91.3% |
| XGBoost + TF-IDF + engineered features | 100.0% |

Latest full-dataset XGBoost run:
```text
Rows: 669,639
TF-IDF features: 2,500
Engineered features: 16
Total features: 2,516
XGBoost - Train Acc: 1.0000, Test Acc: 1.0000
```

Note: the perfect XGBoost score is likely because the dataset labels are strongly rule-derived. The engineered security features expose that labeling pattern very directly, so future portfolio work should explain this limitation and validate on a separate real-world dataset if available.

## Usage
Run the desktop GUI:
```bash
python password_strength_gui.py
```

The GUI trains the Logistic Regression model on startup, then lets you type a password and check its strength.

Run from the command line:
```bash
python password_strength.py
```
Enter a password when prompted. The model will predict its strength.

For a one-off prediction:
```bash
python password_strength.py --password "ExamplePass123!"
```

The script uses Logistic Regression by default because it trains quickly on the full dataset. To run the slower Gradient Boosting model from the notebook:
```bash
python password_strength.py --model gradient-boosting
```

Train the upgraded XGBoost model:
```bash
python train_xgboost.py
```

## Dependencies
pandas, numpy, seaborn, scikit-learn, scipy, xgboost, joblib

## License
Apache License 2.0 - see [LICENSE](LICENSE)
