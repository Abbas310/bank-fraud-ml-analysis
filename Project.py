# Imports
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from tensorflow import keras
from tensorflow.keras import layers

#Load data
df = pd.read_csv("data/Dataset.csv")

#Hierarchical Clustering
numerical_features = df.select_dtypes(include=np.number)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
linked = linkage(scaled_features, method='ward')

df.shape
df.describe()
df.info()
df.head()
df.tail

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Cluster labels
cluster_labels = fcluster(linked, t=2, criterion='maxclust')
y_true_clustering = LabelEncoder().fit_transform(df['TransactionType'])
ari = adjusted_rand_score(y_true_clustering, cluster_labels)
nmi = normalized_mutual_info_score(y_true_clustering, cluster_labels)

#Preprocessing for classification
df_processed = df.copy()
X = df_processed.drop(columns=["TransactionType", "TransactionID"])
y = df_processed["TransactionType"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X.select_dtypes(include=np.number), y_encoded, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#SVM with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train_pca, y_train)

# Predictions and Evaluation
y_pred_svm = svm_classifier.predict(X_test_pca)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_error = 1 - svm_accuracy

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

# Plot Decision Boundary
plt.figure(figsize=(8,6))
xx, yy = np.meshgrid(np.linspace(X_test_pca[:, 0].min()-1, X_test_pca[:, 0].max()+1, 100),
                     np.linspace(X_test_pca[:, 1].min()-1, X_test_pca[:, 1].max()+1, 100))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap='coolwarm')
plt.title('SVM Decision Boundary with PCA-reduced Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Plot ROC Curve for SVM
y_score_svm = svm_classifier.predict_proba(X_test_pca)[:,1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(7,5))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Neural Network
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0)

y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype(int)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_error = 1 - nn_accuracy

cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_))

# Neural Network Training Curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred_rf = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_error = 1 - rf_accuracy

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Random Forest Feature Importance
importances = rf_classifier.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ROC Curve for Random Forest
y_score_rf = rf_classifier.predict_proba(X_test_scaled)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(7,5))
plt.plot(fpr_rf, tpr_rf, color='orange', lw=2, label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

#Final Model Comparison
plt.figure(figsize=(10, 5))
metrics = ['Accuracy', 'Error Rate']
svm_scores = [svm_accuracy, svm_error]
nn_scores = [nn_accuracy, nn_error]
rf_scores = [rf_accuracy, rf_error]

x = np.arange(len(metrics))
width = 0.2

plt.bar(x - width * 1.5, svm_scores, width, label='SVM', color='blue')
plt.bar(x - width / 2, nn_scores, width, label='Neural Network', color='green')
plt.bar(x + width / 2, rf_scores, width, label='Random Forest', color='orange')


plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print Table
print("\nFinal Model Comparison:")
print(f"{'Model':<25}{'Accuracy':<15}{'Error Rate':<15}")
print(f"{'SVM':<25}{svm_accuracy:.4f}{svm_error:.4f}")
print(f"{'Neural Network':<25}{nn_accuracy:.4f}{nn_error:.4f}")
print(f"{'Random Forest':<25}{rf_accuracy:.4f}{rf_error:.4f}")


# Export to CSV
results_df = pd.DataFrame({
    "Model": ["SVM", "Neural Network", "Random Forest"],
    "Accuracy": [svm_accuracy, nn_accuracy, rf_accuracy],
    "Error Rate": [svm_error, nn_error, rf_error]
})

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nModel performance exported to 'model_comparison_results.csv'")