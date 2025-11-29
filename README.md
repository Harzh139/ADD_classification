# 🏡 Property Address Classification  
### **A Hybrid NLP + ML Pipeline for Categorizing Real-Estate Addresses**

![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20TFIDF-orange?style=flat-square)

This project builds a **Machine Learning model** that classifies Indian property addresses into 5 categories:

- **flat**
- **plot**
- **landparcel**
- **commercial unit**
- **others**

It includes **EDA, text preprocessing, feature engineering, TF-IDF vectorization, multiple ML models, cross-validation, and final validation on unseen data**.

---

# 📁 Dataset Overview

We were provided **two datasets**:

| Dataset | Purpose |
|---------|---------|
| **Training Dataset** | Model development & evaluation (EDA, feature engineering, training, CV) |
| **Validation Dataset** | Final performance check on completely unseen data |

---

# 🔍 Exploratory Data Analysis (EDA)

### ✔ Key findings:

- **329 duplicate addresses** → removed  
- Duplicates had **same categories**, so safe to drop
- Category imbalance exists (flats dominate)
- Addresses contain important patterns:
  - Plot numbers  
  - Locality identifiers  
  - Floor/tower information  
  - Keywords like “flat”, “plot”, “villa”
  - Area units (sqft, sq. m, marla, gaj)

These insights shaped the feature engineering process.

---

# 🛠 Feature Engineering

A hybrid of **numeric** and **textual (TF-IDF)** features was used.

---

## 🧮 A. Numeric Features

| Feature | Description |
|--------|--------------|
| `word_count` | Number of words |
| `char_count` | Number of characters |
| `number_present` | Whether digits exist in the address |
| `has_plot` | Contains keyword “plot” |
| `has_flat` | Contains keyword “flat” |
| `has_villa` | Contains keyword “villa” |
| `has_area_info` | Contains area keywords like sqft, sq m, marla, gaj |

These features capture structured signals that TF-IDF alone might miss.

---

## ✍️ B. Text Features (TF-IDF)

We used:

TfidfVectorizer(max_features=3000)

diff
Copy code

This captures:

- Colony/sector/tower names  
- Keywords indicating property type  
- Address semantics  
- Indian real-estate patterns  

Saved as:

models/tfidf.pkl
data/tfidf_matrix.npz

yaml
Copy code

---

# 🤖 Model Training

We tried multiple models:

| Model | Cross-Validation Accuracy |
|--------|--------------------------|
| Logistic Regression | 0.8655 |
| SVM (Linear) | 0.8796 |
| XGBoost | 0.9037 |
| **Random Forest** | ⭐ **0.9079** (Best) |

### ✔ Final Model Chosen  
## **RandomForestClassifier (n_estimators=200)**  
Saved as:

models/RandomForest.pkl
models/label_encoder.pkl

yaml
Copy code

The label encoder ensures correct mapping between numerical labels and category names.

---

# 🔥 Final Validation (on Unseen Dataset)

The validation dataset was processed using:

✔ SAME text cleaning  
✔ SAME numeric feature engineering  
✔ SAME TF-IDF vectorizer (`transform` only, no `fit`)  
✔ Trained RandomForest model  

---

## 🎉 Validation Results (Completely Unseen Data)

### **⭐ Accuracy: 0.9153 (91.53%)**

| Class | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 0 | 0.99 | 0.94 | 0.96 |
| 1 | 0.97 | 0.94 | 0.95 |
| 2 | 0.89 | 0.93 | 0.91 |
| 3 | 0.88 | 0.79 | 0.83 |
| 4 | 0.82 | 0.90 | 0.86 |

### 🔹 Conclusion:
- Strong generalization  
- Minimal overfitting  
- Reliable performance across all classes  
- Production-ready ML pipeline  

---

# 🧠 Why TF-IDF Must Come From Training Only?

- TF-IDF defines a **fixed vocabulary + feature order**  
- The RandomForest model expects this exact structure  
- Re-fitting TF-IDF on validation would break the model  

Thus,  
### **train: fit → transform**  
### **validation: transform only**  
This is the correct ML practice.

---