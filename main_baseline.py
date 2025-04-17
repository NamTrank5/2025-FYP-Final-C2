import os
import cv2
import numpy as np
import pandas as pd
import joblib
from util.img_util import readImageFile
from util.inpaint_util import removeHair
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# === Load data and train Logistic Regression ===
df = pd.read_csv('data/features_with_labels.csv')
feature_cols = [col for col in df.columns if col.startswith("feat_")]
X = df[feature_cols]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Image path ===
image_path = "data/Test_image1.webp"  # Replace with your image name

def extract_features_from_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img_rgb, img_gray = readImageFile(img_path)
    _, _, img_clean = removeHair(img_rgb, img_gray)

    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    compactness = (perimeter ** 2) / (4 * np.pi * area)

    mask = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mean_val = cv2.mean(img_rgb, mask=mask)

    return pd.DataFrame([{
        "feat_area": area,
        "feat_perimeter": perimeter,
        "feat_aspect_ratio": aspect_ratio,
        "feat_extent": extent,
        "feat_compactness": compactness,
        "feat_mean_r": mean_val[2],
        "feat_mean_g": mean_val[1],
        "feat_mean_b": mean_val[0],
    }])

# === Run prediction ===
try:
    new_features = extract_features_from_image(image_path)
    probability = model.predict_proba(new_features)[0][1]
    print(f"🧪 Cancer probability for '{image_path}': {probability:.2%}")

except Exception as e:
    print("❌ Error during processing:", e)
