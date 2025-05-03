# Phishing Detection with Hybrid Feature Selection
# =================================================

# ========== 0. Import Libraries ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    make_scorer
)
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import os

# Configure matplotlib
plt.rcParams['figure.dpi'] = 300
sns.set(style="whitegrid")

# ========== 1. Data Loading & Preprocessing ==========
df = pd.read_csv('data.csv').drop(['id'], axis=1)
X = df.drop(['CLASS_LABEL'], axis=1)
y = df['CLASS_LABEL']

# Initial Feature Importance
model = ExtraTreesClassifier(bootstrap=True, random_state=100)
model.fit(X, y.values.ravel())
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

# Plot Top 50 Feature Importances
top_features = feat_importances.nlargest(50).sort_values()
plt.figure(figsize=(10, 12))
top_features.plot(kind='barh', color='skyblue')
plt.xlabel("Feature Score")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.grid(False)
plt.savefig("Figure 2.png", dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. Hybrid Feature Selection ==========
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
mi_top10 = mi_df.nlargest(10, 'MI_Score')['Feature'].tolist()

rfe_selector = RFE(estimator=ExtraTreesClassifier(), n_features_to_select=10, step=1)
rfe_selector.fit(X, y)
rfe_top10 = X.columns[rfe_selector.support_].tolist()

etc = ExtraTreesClassifier(random_state=42)
etc.fit(X, y)
etc_importances = pd.Series(etc.feature_importances_, index=X.columns)
etc_top10 = etc_importances.nlargest(10).index.tolist()

hybrid_features = list(set(mi_top10 + rfe_top10 + etc_top10))
X_filtered = X[hybrid_features]

# ========== 3. Feature Analysis & Visualization ==========
feature_counts = defaultdict(int)
for method in [mi_top10, rfe_top10, etc_top10]:
    for feature in method:
        feature_counts[feature] += 1

consensus_df = pd.DataFrame({
    'MI': [1 if f in mi_top10 else 0 for f in hybrid_features],
    'RFE': [1 if f in rfe_top10 else 0 for f in hybrid_features],
    'ETC': [1 if f in etc_top10 else 0 for f in hybrid_features]
}, index=hybrid_features)

plt.figure(figsize=(12, 8))
sns.heatmap(consensus_df.T, annot=True, cmap="Blues", fmt='d')
plt.title("Feature Selection Consensus Matrix")
plt.tight_layout()
plt.savefig("Figure 3.png", dpi=300)
plt.show()

print("\n=== Feature Selection Report ===")
print(f"Mutual Information Top 10:\n{mi_top10}")
print(f"\nRFE Top 10:\n{rfe_top10}")
print(f"\nExtra Trees Top 10:\n{etc_top10}")
print(f"\nFinal Hybrid Features ({len(hybrid_features)}):\n{hybrid_features}")

print("\n=== Feature Consensus Count ===")
for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: Selected by {count} method(s)")

# Drop low-importance and highly correlated features
X.drop(X.columns[model.feature_importances_< 0.01], axis=1, inplace=True)
#Features correlation
#get correlations of each features in dataset
corr_matrix = X.corr().abs()
top_corr_features = corr_matrix.index
#plot heat map
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=False, cmap="RdYlGn")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Figure 4.png", dpi=300)
plt.show()

# After dropping irrelevant & correlated features
X_filtered = X[hybrid_features].copy()
X_filtered = X_filtered.drop(columns=[col for col in X_filtered.columns if col not in X.columns])

# ========== 4. Data Normalization ==========

scaler = Normalizer().fit(X_filtered)
X_normalized = scaler.transform(X_filtered)

# ========== 5. Model Evaluation ==========

models = [
    ('ANN', MLPClassifier(random_state=42, max_iter=1000)),
    ('DT', DecisionTreeClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=42)),
    ('LR', LogisticRegression(random_state=42))
]

# Store results
results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
accuracy_res = []
prec_res = []
recall_res = []

from sklearn.model_selection import cross_validate

# Evaluate models
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_normalized, y,
        cv=kfold,
        scoring={ 
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }
    )
    
    results['Model'].append(name)
    results['Accuracy'].append(np.mean(cv_results['test_accuracy']))
    results['Precision'].append(np.mean(cv_results['test_precision']))
    results['Recall'].append(np.mean(cv_results['test_recall']))
    
    # Collect results for boxplots
    accuracy_res.append(cv_results['test_accuracy'])
    prec_res.append(cv_results['test_precision'])
    recall_res.append(cv_results['test_recall'])

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\n=== Model Evaluation Results ===")
print(results_df.round(3))

# Bar Plot of Model Scores
plt.figure(figsize=(12, 6))
results_df.set_index('Model').plot(kind='bar', rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ========== 6. Boxplots of Cross-Validation Metrics ==========
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

# Accuracy
axes[0].boxplot(accuracy_res, labels=[m[0] for m in models])
axes[0].set_title('Accuracy Results')
axes[0].set_ylabel('Accuracy')

# Precision
axes[1].boxplot(prec_res, labels=[m[0] for m in models])
axes[1].set_title('Precision Results')
axes[1].set_ylabel('Precision')

# Recall
axes[2].boxplot(recall_res, labels=[m[0] for m in models])
axes[2].set_title('Recall Results')
axes[2].set_ylabel('Recall')

for ax in axes:
    ax.set_xlabel('Models')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)

plt.subplots_adjust(hspace=0.5)
plt.savefig("Figure 5.png", dpi=300, bbox_inches='tight')
plt.show()
