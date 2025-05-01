# === main_baseline.py ===

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# === Suppress known harmless warnings ===
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')

from sklearn.model_selection import train_test_split
from util.feature_A import get_asymmetry
from util.feature_B import compactness_score
from util.feature_C import get_multicolor_rate
from util.classifier import train_and_evaluate, save_results, save_metrics_and_plot

# === Paths ===
image_dir = "data/images"
mask_dir = "data/lesion_masks"
label_file = "data/metadata.csv"
result_file = "result/results_baseline.csv"
eval_folder = "result/eval"

# === Load metadata ===
df = pd.read_csv(label_file)
malignant_labels = ["MEL", "BCC", "SCC"]
features = []

# === Get all image files ===
image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".png")]

# === Process each image ===
for filename in image_filenames:
    print('Processing:' + filename)
    meta_row = df[df["img_id"] == filename]
    if meta_row.empty:
        continue

    diagnosis = meta_row.iloc[0]["diagnostic"].strip().upper()
    label = 1 if diagnosis in malignant_labels else 0

    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_name = filename.replace(".png", "_mask.png")
    mask_path = os.path.join(mask_dir, mask_name)
    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    try:
        A = get_asymmetry(mask)
        B = compactness_score(mask)
        C = get_multicolor_rate(image, mask, n=5)
        features.append([filename, A, B, C, label])
    except Exception:
        continue

# === Prepare features ===
df_feat = pd.DataFrame(features, columns=["filename", "A", "B", "C", "label"])
before = len(df_feat)
df_feat.dropna(inplace=True)
after = len(df_feat)
print(f"[INFO] Dropped {before - after} rows with missing values.")

X = df_feat[["A", "B", "C"]]
y = df_feat["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train and evaluate ===
clf, y_pred, y_prob, acc, f1, cm = train_and_evaluate(X_train, y_train, X_test, y_test)
os.makedirs("result", exist_ok=True)
os.makedirs("result/eval", exist_ok=True)
save_results(X_test, df_feat.loc[X_test.index, "filename"].values, y_test, y_pred, y_prob, result_file)
save_metrics_and_plot(cm, acc, f1, eval_folder, tag='baseline')

print("\n✅ Done. Results saved in 'result/' folder.")
