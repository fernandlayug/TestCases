import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import xgboost as xgb
import openpyxl
import numpy as np
import shap

# Load your dataset from Excel (replace 'selected_data_1.xlsx' with your Excel file)
data = pd.read_excel('datasets/selected_data_1.xlsx')

# Separate features and target variable
X = data.drop(columns=['Completed'])  # Features
y = data['Completed']  # Target variable

# Encoding categorical variables (example, replace with your actual encoding process)
encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

# Save the encoder to a .pkl file
joblib.dump(encoder, 'model/encoder_2.pkl')

# Keep track of the feature names
feature_names = X_encoded.columns.tolist()

# Split the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Data preparation: Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to a .pkl file
joblib.dump(scaler, 'model/scaler_2.pkl')

# Save the scaler to an Excel file
scaler_data = pd.DataFrame(data={
    'Feature': feature_names,
    'Mean': scaler.mean_,
    'Scale': scaler.scale_
})
scaler_data.to_excel('model/scaler_2.xlsx', index=False)

# Define the XGBoost Classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameters grid for GridSearchCV, including L1 and L2 regularization
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Use the best parameters to train the model
best_xgb = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
best_xgb.fit(X_train_scaled, y_train)

# Save the trained model to a .pkl file
joblib.dump(best_xgb, 'model/best_xgb_model_14.pkl')

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=kf, scoring='roc_auc')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Evaluation on the testing set
y_pred = best_xgb.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC
y_pred_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC Score:", roc_auc)

# Feature importance
feature_importances = best_xgb.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Visualize feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

# SHAP values analysis
explainer = shap.Explainer(best_xgb, X_train_scaled)
shap_values = explainer(X_test_scaled)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# SHAP dependence plots
for feature in feature_names:
    shap.dependence_plot(feature, shap_values.values, X_test, feature_names=feature_names)

# Visualize cross-validation scores
plt.figure(figsize=(8, 6))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores)
plt.xlabel('Fold')
plt.ylabel('ROC AUC')
plt.title('Cross-Validation Scores')
plt.ylim(0.8, 1.0)
plt.show()

# Visualize confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Testing Set')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Visualize the effect of reg_alpha and reg_lambda on cross-validated performance
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=results, x='param_reg_alpha', y='mean_test_score', marker='o')
plt.xlabel('reg_alpha')
plt.ylabel('Mean ROC AUC')
plt.title('Effect of reg_alpha on ROC AUC')

plt.subplot(1, 2, 2)
sns.lineplot(data=results, x='param_reg_lambda', y='mean_test_score', marker='o')
plt.xlabel('reg_lambda')
plt.ylabel('Mean ROC AUC')
plt.title('Effect of reg_lambda on ROC AUC')

plt.tight_layout()
plt.show()
