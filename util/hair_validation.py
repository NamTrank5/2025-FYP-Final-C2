# === util/hair_validation.py ===

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util.hair_feature import detect_hair

# === Paths ===
image_dir = "data/images_mandatory_assignment"
label_file = "data/Hair_Labels.csv"

# === Load manual labels ===
df = pd.read_csv(label_file)
df.columns = ["filename"] + [f"annotator_{i+1}" for i in range(1, len(df.columns))]
df["manual_mean"] = df.iloc[:, 2:7].mean(axis=1)


# === Compute automatic hair scores ===
auto_scores = []

for fname in df["filename"]:
    path = os.path.join(image_dir, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"[!] Could not read: {fname}")
        auto_scores.append(np.nan)
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        score = detect_hair(img)
    except Exception as e:
        print(f"[!] Error for {fname}: {e}")
        score = np.nan
    auto_scores.append(score)

# === Combine and save comparison ===
df["auto_score"] = auto_scores
df.dropna(inplace=True)
df.to_csv("result/hair_validation_comparison.csv", index=False)

# === Plot ===
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df, x="manual_mean", y="auto_score")
plt.title("Manual vs. Automatic Hair Score")
plt.xlabel("Mean Manual Hair Label (0-2)")
plt.ylabel("Auto Hair Ratio (0-1)")
plt.grid(True)
plt.tight_layout()
os.makedirs("result", exist_ok=True)
plt.savefig("result/hair_validation_scatter.png")
plt.close()

# === Print correlation ===
corr = df[["manual_mean", "auto_score"]].corr().iloc[0, 1]
print(f"✅ Correlation between manual and auto hair score: {corr:.3f}")


