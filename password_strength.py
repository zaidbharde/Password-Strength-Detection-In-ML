import warnings
warnings.simplefilter('ignore')

import argparse
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

STRENGTH_MAP = {0: 'Weak', 1: 'Medium', 2: 'Strong'}


def character(inputs):
    characters = []
    for i in inputs:
        characters.append(i)
    return characters


def parse_args():
    parser = argparse.ArgumentParser(description="Train a password strength model and predict passwords.")
    parser.add_argument(
        "--model",
        choices=["logistic", "gradient-boosting"],
        default="logistic",
        help="Model to train. Gradient boosting is slower on the full dataset.",
    )
    parser.add_argument("--password", help="Predict one password and exit.")
    return parser.parse_args()


def train_model(model_name):
    data = pd.read_csv(r"Password Strength.csv", on_bad_lines='skip')
    df = data.dropna()

    x = df['password']
    y = df['strength']

    vec = TfidfVectorizer(tokenizer=character)
    x = vec.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1000)

    if model_name == "gradient-boosting":
        print("Training Gradient Boosting...")
        model = GradientBoostingClassifier()
    else:
        print("Training Logistic Regression...")
        model = LogisticRegression()

    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"{model.__class__.__name__} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    return model, vec


def predict_password(model, vec, password):
    password_vec = vec.transform(np.array([password]))
    prediction = model.predict(password_vec)[0]
    return prediction, STRENGTH_MAP[prediction]


def main():
    args = parse_args()
    model, vec = train_model(args.model)

    if args.password:
        prediction, label = predict_password(model, vec, args.password)
        print(f"Password Strength: {label} ({prediction})")
        return

    while True:
        user_inp = input("\nEnter a password (or 'quit' to exit): ")
        if user_inp.lower() == 'quit':
            break
        prediction, label = predict_password(model, vec, user_inp)
        print(f"Password Strength: {label} ({prediction})")


if __name__ == "__main__":
    main()
