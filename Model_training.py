import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scipy.sparse import load_npz, hstack, csr_matrix
import os

os.makedirs("models", exist_ok=True)


# ---------------------------------------
# LOAD DATA
# ---------------------------------------
df = pd.read_csv("Validation_data/feature_engineered_tfidf.csv")

# ---------------------------------------
# FEATURES & TARGET
# ---------------------------------------
feature_cols = [
    "word_count", "char_count", "number_present",
    "has_plot", "has_flat", "has_villa", "has_area_info"
]
X_numeric = df[feature_cols].values  
X_numeric_sparse = csr_matrix(X_numeric)
X_tfidf = load_npz("Validation_data/tfidf_matrix.npz")
X_final = hstack([X_numeric_sparse, X_tfidf])

X = X_final

# Label encode target
le = LabelEncoder()
y = le.fit_transform(df["categories"])

# Save label encoder
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

# ---------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# MODELS TO TRY (INCLUDING XGBOOST)
# ---------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel="linear"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
}

cv_scores = {}

print("\n========== CROSS VALIDATION ==========\n")

# ---------------------------------------
# CROSS VALIDATION
# ---------------------------------------
for name, model in models.items():
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_scores[name] = scores.mean()
        print(f"{name} CV Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print(f"{name} FAILED DURING CV: {e}")

# ---------------------------------------
# PICK BEST MODEL
# ---------------------------------------
best_model_name = max(cv_scores, key=cv_scores.get)
best_model = models[best_model_name]

print("\n======================================")
print(f" BEST MODEL BASED ON CV: {best_model_name}")
print("======================================\n")

# ---------------------------------------
# TRAIN BEST MODEL
# ---------------------------------------
best_model.fit(X_train, y_train)

# ---------------------------------------
# EVALUATE
# ---------------------------------------
y_pred = best_model.predict(X_test)

print("TEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------
# SAVE BEST MODEL
# ---------------------------------------
pickle.dump(best_model, open(f"models/{best_model_name}.pkl", "wb"))

print(f"\nMODEL SAVED: models/{best_model_name}.pkl")
print("LABEL ENCODER SAVED: models/label_encoder.pkl")
