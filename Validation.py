import pandas as pd
import pickle
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# -------------- CLEAN FUNCTION --------------
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------- LOAD VALIDATION DATA --------------
val_df = pd.read_csv("data\\task_dataset - validation_dataset.csv")

# -------------- APPLY SAME PROCESSING ----------------
val_df["clean_address"] = val_df["property_address"].apply(clean_text)

# Numeric features
val_df["word_count"] = val_df["clean_address"].apply(lambda x: len(x.split()))
val_df["char_count"] = val_df["clean_address"].apply(len)
val_df["number_present"] = val_df["clean_address"].str.contains(r"\d").astype(int)

keywords = ["plot", "flat", "villa"]
for w in keywords:
    val_df[f"has_{w}"] = val_df["clean_address"].str.contains(w).astype(int)

area_keywords = ["sqft", "sq ft", "sq m", "square feet", "marla", "gaj"]
val_df["has_area_info"] = val_df["clean_address"].apply(
    lambda x: 1 if any(a in x for a in area_keywords) else 0
)

feature_cols = [
    "word_count", "char_count", "number_present",
    "has_plot", "has_flat", "has_villa", "has_area_info"
]

# -------------- LOAD TF-IDF --------------
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
X_val_tfidf = tfidf.transform(val_df["clean_address"])

# Numeric → sparse
X_val_numeric = csr_matrix(val_df[feature_cols].values)

# Combine
X_val = hstack([X_val_numeric, X_val_tfidf])

# -------------- LOAD MODEL + LABEL ENCODER --------------
model = pickle.load(open("models/RandomForest.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

# True labels
y_true = le.transform(val_df["categories"])

# -------------- PREDICT --------------
y_pred = model.predict(X_val)

# -------------- EVALUATE --------------
print("\nValidation Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
