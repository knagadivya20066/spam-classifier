"""
train.py
--------
Trains a Multinomial Naive Bayes spam classifier on the SMS Spam Collection
Dataset and saves the trained model + vectorizer to disk using pickle.

Usage
-----
    python train.py

Output
------
    models/spam_model.pkl        -> trained MultinomialNB model
    models/tfidf_vectorizer.pkl  -> fitted TF-IDF vectorizer
    A full evaluation report printed to the console (accuracy, precision,
    recall, F1-score, and confusion matrix).
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocess import clean_text

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "spam.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
RANDOM_STATE = 42          # Fixed seed -> reproducible train/test splits
TEST_SIZE = 0.2            # 80% train / 20% test split


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the SMS Spam Collection Dataset from a CSV file.

    Expected columns: 'label' ('ham'/'spam') and 'message' (raw SMS text).
    """
    df = pd.read_csv(path, encoding="latin-1")

    # Keep only the two columns we need, in case the raw file (as downloaded
    # from Kaggle/UCI) contains extra unnamed columns.
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    # Drop any rows with missing values and duplicate messages
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df.reset_index(drop=True)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to every message and encode labels as 0/1."""
    print("Cleaning text (lowercasing, removing punctuation/stopwords, tokenizing)...")
    df["cleaned_message"] = df["message"].apply(clean_text)

    # Encode labels: ham -> 0, spam -> 1 (required for sklearn metrics)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def train_model(X_train, y_train) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> None:
    """Print accuracy, precision, recall, F1-score and confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print("                 Predicted Ham   Predicted Spam")
    print(f"Actual Ham         {cm[0][0]:<15}{cm[0][1]}")
    print(f"Actual Spam        {cm[1][0]:<15}{cm[1][1]}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("=" * 50)


def save_artifacts(model: MultinomialNB, vectorizer: TfidfVectorizer) -> None:
    """Persist the trained model and vectorizer to disk using pickle."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to      : {MODEL_PATH}")
    print(f"Vectorizer saved to : {VECTORIZER_PATH}")


def main():
    # 1. Load raw data
    print(f"Loading dataset from '{DATASET_PATH}'...")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} messages "
          f"({(df['label'] == 'ham').sum()} ham / {(df['label'] == 'spam').sum()} spam)")

    # 2. Preprocess text
    df = preprocess_dataset(df)

    # 3. Split into train/test sets BEFORE vectorizing to avoid data leakage
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_message"],
        df["label_num"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],  # preserve the ham/spam ratio in both splits
    )

    # 4. TF-IDF Vectorization
    #    - fit_transform on the TRAINING data only
    #    - transform (never fit) on the TEST data, so the model never "sees"
    #      test-set vocabulary during training
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 5. Train the Multinomial Naive Bayes model
    print("Training Multinomial Naive Bayes classifier...")
    model = train_model(X_train, y_train)

    # 6. Evaluate performance on the held-out test set
    evaluate_model(model, X_test, y_test)

    # 7. Save the trained model + vectorizer for later use in predict.py
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()
