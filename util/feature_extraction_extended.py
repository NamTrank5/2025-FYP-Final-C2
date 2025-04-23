def extract_features_from_folder_nohair(metadata_csv, image_folder, output_csv):
    import os
    import cv2
    import numpy as np
    import pandas as pd
    from util.img_util import readImageFile
    from util.inpaint_util import removeHair

    metadata = pd.read_csv(metadata_csv)
    metadata["label"] = metadata["biopsed"].astype(int)

    features = []

    for idx, row in metadata.iterrows():
        img_id = row["img_id"]
        label = row["label"]
        img_path = os.path.join(image_folder, img_id)

        if not os.path.exists(img_path):
            print(f"Missing: {img_id}")
            continue

        try:
            img_rgb, img_gray = readImageFile(img_path)
            _, _, img_clean = removeHair(img_rgb, img_gray)

            img_clean_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            compactness = (perimeter ** 2) / (4 * np.pi * area)

            mask = np.zeros(img_gray.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean_val = cv2.mean(img_clean, mask=mask)

            features.append({
                "img_id": img_id,
                "feat_area": area,
                "feat_perimeter": perimeter,
                "feat_aspect_ratio": aspect_ratio,
                "feat_extent": extent,
                "feat_compactness": compactness,
                "feat_mean_r": mean_val[2],
                "feat_mean_g": mean_val[1],
                "feat_mean_b": mean_val[0],
                "label": label
            })

        except Exception as e:
            print(f"Error on {img_id}: {e}")

    df_out = pd.DataFrame(features)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved feature CSV: {output_csv}")
