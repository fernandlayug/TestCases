import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.impute import SimpleImputer
import numpy as np

# Load data from Excel file
df = pd.read_excel('datasets/balanced_data_1.xlsx')

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Completed'])
y = df['Completed']

# Handle missing values by imputing with the mean (can be adjusted to other strategies if needed)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Function to perform feature selection with cross-validation
def select_features_cv(X, y, k=20, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    scores = np.zeros(X.shape[1])
    fold_scores = []  # List to store scores of each fold

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_train, y_train)
        fold_scores.append(selector.scores_)
        scores += selector.scores_

    scores /= n_splits
    selector.scores_ = scores
    selected_features_indices = np.argsort(scores)[-k:]
    selected_features = X.columns[selected_features_indices]
    
    return X.iloc[:, selected_features_indices], selected_features, selector, fold_scores

# Specify the number of features you want to select (k)
k_features = 20

# Perform feature selection with cross-validation
X_selected, selected_features, selector, fold_scores = select_features_cv(X, y, k=k_features)

# Convert fold_scores to DataFrame for easier plotting
fold_scores_df = pd.DataFrame(fold_scores, columns=X.columns)

# Save the selected features to an Excel file
selected_features_df = pd.DataFrame({
    'Feature': selected_features,
    'Average ANOVA F-value': selector.scores_[np.argsort(selector.scores_)[-k_features:]]
})
selected_features_df.to_excel('datasets/selected_features_1.xlsx', index=False)

# Save the dataset with only the selected features and target variable to an Excel file
selected_data = pd.concat([X_selected, y], axis=1)
selected_data.to_excel('datasets/selected_data_1.xlsx', index=False)

# Visualization 1: Box plot of cross-validation scores for all features
plt.figure(figsize=(14, 10))
sns.boxplot(data=fold_scores_df)
plt.xlabel('Features')
plt.ylabel('ANOVA F-value')
plt.title('Cross-Validation Scores for All Features')
plt.xticks(rotation=90)
plt.show()

# Visualization 2: Heatmap of feature scores across folds
plt.figure(figsize=(14, 10))
sns.heatmap(fold_scores_df, annot=True, cmap='coolwarm', center=0)
plt.xlabel('Features')
plt.ylabel('Folds')
plt.title('Heatmap of Feature Scores Across Folds')
plt.show()

# Visualization 3: Box plot for selected features only
plt.figure(figsize=(12, 8))
sns.boxplot(data=fold_scores_df[selected_features])
plt.xlabel('Features')
plt.ylabel('ANOVA F-value')
plt.title('Cross-Validation Scores for Selected Features')
plt.xticks(rotation=45)
plt.show()

# Visualization 4: Average scores of the selected features
plt.figure(figsize=(10, 6))
plt.barh(selected_features, selector.scores_[np.argsort(selector.scores_)[-k_features:]], color='skyblue')
plt.xlabel('ANOVA F-value')
plt.title('Top {} Selected Features (Average Scores)'.format(k_features))
plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
plt.show()

# Find the best regularization parameter using LassoCV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_selected, y)

# Print the best alpha found by LassoCV
print("Best alpha found by LassoCV:", lasso_cv.alpha_)

# Apply Lasso Regression with the best alpha
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_selected, y)

# Print Lasso coefficients
print("Lasso coefficients:", lasso.coef_)

# Cross-validation with Logistic Regression
logreg = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(logreg, X_selected, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train and evaluate Logistic Regression on validation set
logreg.fit(X_train, y_train)
train_score = logreg.score(X_train, y_train)
val_score = logreg.score(X_val, y_val)

print("Training set score:", train_score)
print("Validation set score:", val_score)

# If there is a significant drop in performance from training to validation, there might be overfitting
