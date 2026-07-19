# Checking Password Strength using Machine Learning

**Author:** Zaid Bharde

This project builds a machine learning model to predict password strength (weak, medium, strong) using character-level TF-IDF features. It uses Logistic Regression and Gradient Boosting classifiers.

## Dataset
- **Password** - 670k+ unique passwords collected online
- **Strength** - 0 (weak), 1 (medium), 2 (strong)

## Feature Engineering
Passwords are converted to numerical features using character-level TF-IDF vectorization.

## Models & Performance
| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~81.8% |
| Gradient Boosting | ~91.3% |

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

## Dependencies
pandas, numpy, seaborn, scikit-learn

## License
Apache License 2.0 - see [LICENSE](LICENSE)
