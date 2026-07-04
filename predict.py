"""
predict.py
----------
Loads the trained spam classifier and TF-IDF vectorizer, then predicts
whether a custom, user-supplied message is "ham" (not spam) or "spam".

Usage
-----
    # Interactive mode - type messages one at a time
    python predict.py

    # Single message from the command line
    python predict.py "Congratulations! You have WON a free prize, call now!"
"""

import os
import pickle
import sys

from preprocess import clean_text

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def load_artifacts():
    """Load the pickled model and vectorizer from disk."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        raise FileNotFoundError(
            "Trained model files not found. Please run 'python train.py' "
            "first to train and save the model."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_message(message: str, model, vectorizer) -> dict:
    """
    Predict whether a single message is spam or ham.

    Returns a dictionary containing the predicted label and the model's
    confidence (probability) for that prediction, which is more informative
    for a user than a bare label.
    """
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    label = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    confidence = probabilities[prediction]

    return {"label": label, "confidence": confidence}


def run_interactive_mode(model, vectorizer):
    """Continuously prompt the user for messages until they type 'exit'."""
    print("\nSpam Email/SMS Classifier - Interactive Mode")
    print("Type a message to classify it, or type 'exit' to quit.\n")

    while True:
        message = input("Enter a message: ").strip()

        if message.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not message:
            print("Please enter a non-empty message.\n")
            continue

        result = predict_message(message, model, vectorizer)
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}\n")


def main():
    model, vectorizer = load_artifacts()

    # If a message was passed as a command-line argument, classify it once
    # and exit. Otherwise, drop into interactive mode.
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        result = predict_message(message, model, vectorizer)
        print(f"Message    : {message}")
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}")
    else:
        run_interactive_mode(model, vectorizer)


if __name__ == "__main__":
    main()
