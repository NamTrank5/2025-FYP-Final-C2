# === main.py ===
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from util.feature_extraction import extract_features_from_folder
import matplotlib.pyplot as plt
import seaborn as sns

# === Optional Feature Extraction ===
metadata_path = "data/metadata.csv"
images_path = "data/images"
output_csv = "data/features_with_labels.csv"

# Uncomment the line below if you want to re-run feature extraction (slow)
# extract_features_from_folder(metadata_path, images_path, output_csv)

# === Load extracted features ===
df = pd.read_csv('data/features_with_labels.csv')
feature_cols = [col for col in df.columns if col.startswith("feat_")]
X = df[feature_cols]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train multiple models ===
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

results = []
os.makedirs("result/confusion_matrices", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🔍 {name}")
    print(f"✅ Accuracy: {acc:.2f}")
    print("🧾 Confusion Matrix:")
    print(cm)

    # Save numeric results
    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "ConfusionMatrix": str(cm.tolist())
    })

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"])
    cm_df.to_csv(f"result/confusion_matrices/{name.replace(' ', '_')}_cm.csv")

    # Save confusion matrix as heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"result/confusion_matrices/{name.replace(' ', '_')}_cm.png")
    plt.close()

# Save final model comparison table
results_df = pd.DataFrame(results)
results_df.to_csv("result/model_comparison.csv", index=False)
print("\n✅ Model results saved to: result/model_comparison.csv")

