# 📧 Spam Email Classifier

A complete, end-to-end machine learning project that classifies SMS/email messages as **Spam** or **Ham (Not Spam)** using classic Natural Language Processing techniques and a Multinomial Naive Bayes classifier.

Built as a college internship / portfolio project, but structured the way a production ML repo would be — clean modular code, reproducible training, and a saved model ready for inference.

---

## 📌 Project Overview

Spam detection is one of the most well-known applied NLP problems, and it's a great way to demonstrate a full ML workflow: from raw, messy text data to a saved, reusable model.

This project:
- Loads the **SMS Spam Collection Dataset** (5,169 unique messages after cleaning)
- Cleans and normalizes the text (lowercasing, punctuation removal, stopword removal, tokenization)
- Converts text into numerical features using **TF-IDF Vectorization**
- Trains a **Multinomial Naive Bayes** classifier — a standard, fast, and highly effective algorithm for text classification
- Evaluates the model with accuracy, precision, recall, F1-score, and a confusion matrix
- Saves the trained model with `pickle` so it can be reused without retraining
- Lets you classify your **own custom messages** from the command line

---

## ✨ Features

- ✅ Clean, well-commented, beginner-friendly code
- ✅ Modular design (`preprocess.py` shared between training & inference — no logic duplication)
- ✅ Proper train/test split with **no data leakage** (TF-IDF fit only on training data)
- ✅ Full evaluation suite: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- ✅ Interactive CLI for testing your own messages
- ✅ Trained model persisted with `pickle` for instant reuse
- ✅ Companion Jupyter notebook with EDA and visualizations
- ✅ Clear, professional project structure ready for GitHub

---

## 🛠️ Technologies

| Category            | Tools / Libraries                     |
|---------------------|----------------------------------------|
| Language             | Python 3.8+                           |
| Data Handling        | pandas, numpy                         |
| NLP / Preprocessing  | NLTK (stopwords, tokenization)        |
| Feature Extraction   | scikit-learn (TF-IDF Vectorizer)      |
| Machine Learning     | scikit-learn (Multinomial Naive Bayes)|
| Visualization        | matplotlib, seaborn                   |
| Model Persistence    | pickle                                |
| Notebook             | Jupyter                               |

---

## 📂 Dataset

This project uses the **[SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)**, a public dataset of 5,572 real SMS messages labeled as `ham` (legitimate) or `spam`.

| Stat                         | Value |
|-------------------------------|-------|
| Total messages (raw)          | 5,572 |
| Unique messages (after cleaning) | 5,169 |
| Ham messages                  | ~4,516 (87%) |
| Spam messages                 | ~653 (13%) |

### Download Instructions

The cleaned dataset is already included at `dataset/spam.csv` in this repository, so **no download is required to run the project as-is**.

If you'd like to fetch it yourself from the original source:

1. **Option A — UCI Machine Learning Repository**
   Visit: https://archive.ics.uci.edu/dataset/228/sms+spam+collection and download `smsspamcollection.zip`, then extract the `SMSSpamCollection` file.

2. **Option B — Kaggle**
   Search "SMS Spam Collection Dataset" on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and download `spam.csv`.

3. Place the file inside the `dataset/` folder and rename it to `spam.csv` with two columns: `label` (`ham`/`spam`) and `message` (the raw text).

> The dataset is a single CSV file with two relevant columns — `v1` (label) and `v2` (message) in the original Kaggle format, or `label`/`message` in the cleaned format used by this project. `train.py` automatically handles either column naming.

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Spam_Email_Classifier.git
   cd Spam_Email_Classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **NLTK data (auto-downloaded on first run)**
   The first time you run `train.py` or `predict.py`, the required NLTK resources (`stopwords`, `punkt`) will be downloaded automatically. No manual step needed.

---

## 🚀 Usage

### 1. Train the model

```bash
python train.py
```

This will:
- Load and clean the dataset
- Split it into training (80%) and testing (20%) sets
- Vectorize the text using TF-IDF
- Train the Multinomial Naive Bayes model
- Print the full evaluation report to the console
- Save `models/spam_model.pkl` and `models/tfidf_vectorizer.pkl`

### 2. Predict on custom messages

**Interactive mode** — type as many messages as you like:
```bash
python predict.py
```
```
Enter a message: Congratulations! You have WON a free prize, call now!
Prediction : SPAM
Confidence : 98.25%

Enter a message: Hey, are we still meeting for lunch tomorrow?
Prediction : HAM (Not Spam)
Confidence : 99.64%
```

**One-off prediction** from the command line:
```bash
python predict.py "URGENT! Your mobile number has won 5000 pounds, text CLAIM to 8712 now"
```

### 3. Explore the notebook

```bash
jupyter notebook notebook.ipynb
```
The notebook walks through the same pipeline with added exploratory data analysis and charts (class distribution, message length, confusion matrix heatmap).

---

## 📊 Results

Evaluated on a held-out test set (20% of the data, 1,034 messages), using a fixed random seed for reproducibility:

| Metric      | Score  |
|-------------|--------|
| **Accuracy**  | **97.00%** |
| **Precision** | **99.02%** |
| **Recall**    | **77.10%** |
| **F1-Score**  | **86.70%** |

### Confusion Matrix

|                  | Predicted Ham | Predicted Spam |
|------------------|:-------------:|:--------------:|
| **Actual Ham**   | 902           | 1               |
| **Actual Spam**  | 30            | 101             |

**Interpretation:**
- The model almost never misclassifies a real message as spam (only 1 false positive out of 903 ham messages) — this matters a lot in practice, since flagging a genuine message as spam is usually more costly than missing a spam message.
- Precision is very high (99%), meaning that when the model *does* flag something as spam, it's almost always right.
- Recall is comparatively lower (77%), meaning some spam messages slip through undetected. This precision/recall trade-off is typical and expected for Naive Bayes on an imbalanced dataset (~13% spam), and could be tuned further (see below).

---

## 🚧 Future Improvements

- [ ] Experiment with other algorithms (Logistic Regression, SVM, Random Forest) and compare performance
- [ ] Use `class_weight` balancing or resampling (SMOTE) to improve recall on the minority (spam) class
- [ ] Add n-gram features (bigrams/trigrams) to the TF-IDF vectorizer to capture phrases like "free entry" or "text now"
- [ ] Add stemming/lemmatization to further normalize the vocabulary
- [ ] Build a simple web interface (Flask/Streamlit) for live predictions
- [ ] Add cross-validation instead of a single train/test split for more robust metrics
- [ ] Package the model as a REST API for integration into other applications
- [ ] Add unit tests for `preprocess.py`, `train.py`, and `predict.py`

---

## 📁 Project Structure

```
Spam_Email_Classifier/
│
├── dataset/
│   └── spam.csv                # SMS Spam Collection Dataset (label, message)
├── models/
│   ├── spam_model.pkl           # Trained Multinomial Naive Bayes model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── screenshots/
│   ├── class_distribution.png
│   ├── message_length_distribution.png
│   └── confusion_matrix.png
├── preprocess.py                 # Shared text-cleaning logic (used by train & predict)
├── train.py                       # Training script
├── predict.py                     # Prediction / inference script
├── notebook.ipynb                 # EDA + step-by-step walkthrough
├── requirements.txt
└── README.md
```

---

## 📷 Screenshots

All screenshots live in the `screenshots/` folder:

| Screenshot | Description |
|---|---|
| `train_output.png` | Console output of `python train.py` — full evaluation report |
| `predict_interactive.png` | `python predict.py` interactive mode classifying sample messages |
| `class_distribution.png` | Bar chart of ham vs spam message counts |
| `message_length_distribution.png` | Histogram comparing message lengths by label |
| `confusion_matrix.png` | Heatmap of model predictions vs actual labels |

**Class Distribution**

![Class Distribution](screenshots/class_distribution.png)

**Confusion Matrix**

![Confusion Matrix](screenshots/confusion_matrix.png)

**Training Output**

![Train Output](screenshots/train_output.png)

**Interactive Prediction**

![Predict Interactive](screenshots/predict_interactive.png)

---

## 👤 Author

Built as an internship / portfolio project to demonstrate a complete, practical machine learning workflow — from raw text data to a deployable, reusable model.

## 📄 License

This project is open-source and available under the MIT License. The dataset is provided by the UCI Machine Learning Repository for research/educational use.
