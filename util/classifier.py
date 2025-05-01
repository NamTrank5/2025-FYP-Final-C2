from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import os


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, y_pred, y_prob, acc, f1, cm


def save_results(X_test, filenames, y_test, y_pred, y_prob, result_path):
    df = X_test.copy()
    df["filename"] = filenames
    df["label"] = y_test.values
    df["prediction"] = y_pred
    df["probability"] = y_prob
    df = df[["filename", "probability", "label", "prediction"]]
    df.to_csv(result_path, index=False)

def save_metrics_and_plot(cm, acc, f1, output_folder, tag="baseline"):
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(output_folder, exist_ok=True)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {tag.capitalize()}")
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_{tag}.png"))
    plt.close()

    # Save metrics
    with open(os.path.join(output_folder, f"metrics_{tag}.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
