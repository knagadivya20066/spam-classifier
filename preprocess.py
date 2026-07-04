"""
preprocess.py
--------------
Shared text-preprocessing utilities for the Spam Email Classifier project.

Keeping this logic in one file (instead of copy-pasting it into train.py
and predict.py) guarantees that a message is cleaned in *exactly* the same
way at both training time and prediction time. This consistency is critical
for any NLP pipeline -- if the cleaning steps ever drift apart, the model
will silently perform worse on real predictions.
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK resource setup
# ---------------------------------------------------------------------------
# NLTK needs a few small data packages (stopword lists, tokenizer models)
# downloaded before they can be used. We attempt the download quietly and
# ignore failures caused by an already-satisfied cache, so this file can be
# safely imported multiple times without re-downloading anything.
def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()

# Load the English stopword list once at import time (fast lookups later).
STOPWORDS = set(stopwords.words("english"))

# Pre-build a translation table that maps every punctuation character to None.
# This is the fastest standard way to strip punctuation from a string in Python.
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(message: str) -> str:
    """
    Clean and normalize a single raw text message.

    Steps performed (in order):
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenize into individual words
        4. Remove stopwords (e.g. "the", "is", "at")
        5. Re-join the remaining tokens into a single cleaned string

    Parameters
    ----------
    message : str
        The raw, unprocessed text message (SMS / email body).

    Returns
    -------
    str
        The cleaned message, ready to be fed into a vectorizer (e.g. TF-IDF).
    """
    # Guard against non-string / missing input
    if not isinstance(message, str):
        return ""

    # 1. Lowercase everything so "FREE" and "free" are treated the same
    message = message.lower()

    # 2. Strip punctuation (commas, periods, !!!, etc.)
    message = message.translate(_PUNCTUATION_TABLE)

    # 3. Tokenize the message into a list of words
    tokens = word_tokenize(message)

    # 4. Remove stopwords and any leftover non-alphabetic tokens (numbers, etc.)
    cleaned_tokens = [
        word for word in tokens
        if word not in STOPWORDS and word.isalpha()
    ]

    # 5. Re-join tokens back into a single string (TF-IDF expects raw text,
    #    not a list of tokens)
    return " ".join(cleaned_tokens)
