import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from util.feature_extraction_extended import extract_features_from_folder_nohair

metadata_path = "data/metadata.csv"
images_path = "data/images"
output_csv = "data/features_with_labels_nohair.csv"

# Uncomment the line below if you want to re-run feature extraction (slow)
extract_features_from_folder_nohair(metadata_path, images_path, output_csv)

# === Load Features ===
df = pd.read_csv(output_csv)
feature_cols = [col for col in df.columns if col.startswith("feat_")]
X = df[feature_cols]
y = df["label"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Logistic Regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("📊 Logistic Regression (Baseline - With Hair Removal)")
print(f"✅ Accuracy: {accuracy:.2%}")
print("🧾 Confusion Matrix:")
print(conf_matrix)

# === Save Results ===
os.makedirs("result", exist_ok=True)
pd.DataFrame({
    "Accuracy": [accuracy],
    "TN": [conf_matrix[0][0]],
    "FP": [conf_matrix[0][1]],
    "FN": [conf_matrix[1][0]],
    "TP": [conf_matrix[1][1]]
}).to_csv("result/extended_result.csv", index=False)