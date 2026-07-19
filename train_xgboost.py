import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_engineering import FEATURE_NAMES, build_feature_matrix, load_common_passwords


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Password Strength.csv"
MODEL_PATH = BASE_DIR / "models" / "password_strength_xgboost.joblib"


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost password strength model with TF-IDF plus engineered features.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the password CSV dataset.")
    parser.add_argument("--sample", type=int, help="Optional row sample for quick experiments.")
    parser.add_argument("--max-features", type=int, default=2500, help="Maximum character TF-IDF features.")
    parser.add_argument("--estimators", type=int, default=180, help="Number of XGBoost trees.")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Path for the trained model artifact.")
    return parser.parse_args()


def load_dataset(path, sample=None):
    data = pd.read_csv(path, on_bad_lines="skip").dropna()
    if sample and sample < len(data):
        data = data.sample(n=sample, random_state=1000)
    return data["password"].astype(str), data["strength"].astype(int)


def build_combined_features(passwords, vectorizer=None):
    common_passwords = load_common_passwords()
    engineered = build_feature_matrix(passwords, common_passwords)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=2500)
        tfidf = vectorizer.fit_transform(passwords)
    else:
        tfidf = vectorizer.transform(passwords)

    return hstack([tfidf, engineered], format="csr"), vectorizer


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    passwords, labels = load_dataset(args.data, args.sample)
    print(f"Rows: {len(passwords):,}")

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        passwords,
        labels,
        test_size=0.2,
        random_state=1000,
        stratify=labels,
    )

    print("Building TF-IDF + engineered features...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=args.max_features)
    common_passwords = load_common_passwords()
    x_train_tfidf = vectorizer.fit_transform(x_train_raw)
    x_test_tfidf = vectorizer.transform(x_test_raw)
    x_train_eng = build_feature_matrix(x_train_raw, common_passwords)
    x_test_eng = build_feature_matrix(x_test_raw, common_passwords)
    x_train = hstack([x_train_tfidf, x_train_eng], format="csr")
    x_test = hstack([x_test_tfidf, x_test_eng], format="csr")

    print(f"TF-IDF features: {len(vectorizer.get_feature_names_out()):,}")
    print(f"Engineered features: {len(FEATURE_NAMES)}")
    print(f"Total features: {x_train.shape[1]:,}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.estimators,
        max_depth=6,
        learning_rate=0.10,
        subsample=0.9,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=1000,
    )

    print("Training XGBoost...")
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"XGBoost - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "feature_names": list(vectorizer.get_feature_names_out()) + FEATURE_NAMES,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "rows": len(passwords),
    }
    joblib.dump(artifact, output_path)
    print(f"Saved model artifact: {output_path}")


if __name__ == "__main__":
    main()
