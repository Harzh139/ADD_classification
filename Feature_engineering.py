import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import save_npz


df = pd.read_csv("data/cleaned_dataset.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_address"] = df["property_address"].apply(clean_text)

# Basic numeric features
df["word_count"] = df["clean_address"].apply(lambda x: len(x.split()))
df["char_count"] = df["clean_address"].apply(len)
df["number_present"] = df["clean_address"].str.contains(r"\d").astype(int)

# Keyword flags
keywords = ["plot", "flat", "villa"]

for w in keywords:
    df[f"has_{w}"] = df["clean_address"].str.contains(w).astype(int)

# Area feature
area_keywords = ["sqft", "sq ft", "sq m", "square feet", "marla", "gaj"]
df["has_area_info"] = df["clean_address"].apply(
    lambda x: 1 if any(a in x for a in area_keywords) else 0
)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["clean_address"])

pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))

save_npz("data/tfidf_matrix.npz", X_tfidf)
# Save data for model training
df.to_csv("data/feature_engineered_tfidf.csv", index=False)

print("Feature engineering (with TF-IDF) done.")
