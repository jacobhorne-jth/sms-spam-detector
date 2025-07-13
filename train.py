import pandas as pd
import html
import re
import string
import numpy as np
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils import resample
import joblib


spam_keywords = set([
    "free", "win", "winner", "prize", "cash", "urgent", "offer", "claim",
    "gift", "congratulations", "reward", "discount", "bonus", "exclusive",
    "deal", "save", "promotion", "apply", "now", "limited", "credit",
    "loan", "mortgage", "finance", "income", "earn", "investment", "money",
    "call", "reply", "click", "subscribe", "buy", "order", "alert",
    "important", "immediately", "verify", "security", "account", "warning",
    "recruit", "paid", "contact", "recruiter"
])


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.lower()
    text = re.sub(r"\+?\d[\d\-\s]{6,}\d", "phone_number", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def extract_features(texts):
    """Return a 2D numpy array of custom features per text"""
    features = []
    for text in texts:
        orig = text if isinstance(text, str) else ""
        phone_flag = int(bool(re.search(r"\+?\d[\d\-\s]{6,}\d", orig)))
        money_flag = int(bool(re.search(r"[£$€]", orig)))
        keyword_count = sum(word in orig.lower() for word in spam_keywords)
        num_upper = sum(1 for c in orig if c.isupper())
        upper_ratio = num_upper / len(orig) if len(orig) > 0 else 0
        features.append([phone_flag, money_flag, keyword_count, upper_ratio])
    return np.array(features)


def load_datasets():
    df_original = pd.read_csv("datasets/spam.csv", encoding="Windows-1252", usecols=[0,1], names=["label", "message"], skiprows=1)
    df_synth1 = pd.read_csv("datasets/synthetic_spam_messages.csv")
    df_synth2 = pd.read_csv("datasets/synthetic_smishing_messages.csv")
    df_synth3 = pd.read_csv("datasets/synthetic_reply_with_y_spam_v2.csv")
    df_other = pd.read_csv("datasets/Dataset_5971.csv", encoding="utf-8", usecols=[0,1], names=["label", "message"], skiprows=1)
    df_other["label"] = df_other["label"].replace("Smishing", "spam")
    df_all = pd.concat([df_original, df_synth1, df_synth2, df_synth3, df_other], ignore_index=True)
    return df_all


def preprocess_data(df):
    df["clean_message"] = df["message"].apply(clean_text)
    y = (df["label"].str.lower() == "spam").astype(int)
    return df["message"], df["clean_message"], y  # Keep raw + clean


def train_model(X_raw_train, X_clean_train, y_train):
    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True
    )
    # Fit TF-IDF on clean text
    X_clean_tfidf = tfidf.fit_transform(X_clean_train)

    # Upsample spam
    df_train = pd.DataFrame({"text_raw": X_raw_train, "text_clean": X_clean_train, "label": y_train})
    df_majority = df_train[df_train.label == 0]
    df_minority = df_train[df_train.label == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X_upsampled_raw = df_upsampled["text_raw"]
    X_upsampled_clean = df_upsampled["text_clean"]
    y_upsampled = df_upsampled["label"]

    # Extract features for upsampled raw text
    custom_feats = extract_features(X_upsampled_raw)

    # TF-IDF features for upsampled clean text
    X_upsampled_tfidf = tfidf.transform(X_upsampled_clean)

    # Combine features horizontally
    X_train_combined = hstack([X_upsampled_tfidf, custom_feats])

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_combined, y_upsampled)

    return model, tfidf


def evaluate_model(model, vectorizer, X_raw_test, X_clean_test, y_test):
    X_test_tfidf = vectorizer.transform(X_clean_test)
    custom_feats = extract_features(X_raw_test)
    X_test_combined = hstack([X_test_tfidf, custom_feats])
    y_pred = model.predict(X_test_combined)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def save_model(model, vectorizer):
    joblib.dump(model, "model/spam_detector_model.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")


def load_model():
    model = joblib.load("model/spam_detector_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


def predict_message(model, vectorizer, text):
    cleaned = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    custom_feats = extract_features([text])
    combined = hstack([tfidf_vec, custom_feats])
    proba = model.predict_proba(combined)[0]
    spam_prob = proba[1]
    pred = int(spam_prob >= 0.4)
    return pred, spam_prob


if __name__ == "__main__":
    df = load_datasets()
    X_raw, X_clean, y = preprocess_data(df)
    X_train_raw, X_test_raw, X_train_clean, X_test_clean, y_train, y_test = train_test_split(
        X_raw, X_clean, y, test_size=0.2, random_state=42
    )

    model, vectorizer = train_model(X_train_raw, X_train_clean, y_train)
    evaluate_model(model, vectorizer, X_test_raw, X_test_clean, y_test)
    save_model(model, vectorizer)

    example_text = "You have won £1000 cash! Call to claim your prize."
    result, prob = predict_message(model, vectorizer, example_text)
    if result:
        print(f"🚨 This message is SPAM. (Spam probability: {prob:.2f})")
    else:
        print(f"✅ This message is HAM. (Spam probability: {prob:.2f})")
