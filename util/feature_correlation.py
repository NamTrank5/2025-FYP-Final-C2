import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("result/features_extended.csv")

corr = df[["A", "B", "C", "H", "label"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("result/eval_extended/feature_correlation.png")
plt.close()

print("✅ Correlation matrix saved as 'feature_correlation.png'")
