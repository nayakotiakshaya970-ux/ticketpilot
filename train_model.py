import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
df = pd.read_csv("data/support_tickets.csv")
print(f"Loaded {len(df)} tickets\n")
print(df["category"].value_counts(), "\n")

X = df["ticket_text"]
y_cat = df["category"]
y_urg = df["urgency"]

# 2. Train/test split (80/20)
X_train, X_test, cat_train, cat_test, urg_train, urg_test = train_test_split(
    X, y_cat, y_urg, test_size=0.2, random_state=42, stratify=y_cat
)

# 3. Text -> TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Category model
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train_vec, cat_train)
cat_pred = category_model.predict(X_test_vec)
cat_acc = accuracy_score(cat_test, cat_pred)

print("=== CATEGORY CLASSIFICATION ===")
print(f"Accuracy: {cat_acc:.2%}")
print(classification_report(cat_test, cat_pred))

# 5. Urgency model
urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_train_vec, urg_train)
urg_pred = urgency_model.predict(X_test_vec)
urg_acc = accuracy_score(urg_test, urg_pred)

print("=== URGENCY CLASSIFICATION ===")
print(f"Accuracy: {urg_acc:.2%}")
print(classification_report(urg_test, urg_pred))

# 6. Save models and metrics
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(category_model, "models/category_model.pkl")
joblib.dump(urgency_model, "models/urgency_model.pkl")

with open("models/metrics.txt", "w") as f:
    f.write(f"Category accuracy: {cat_acc:.2%}\n")
    f.write(f"Urgency accuracy: {urg_acc:.2%}\n")

print("Models saved to models/ folder")