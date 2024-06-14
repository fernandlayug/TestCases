import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Step 1: Load the data from Excel into a Pandas DataFrame
excel_file = 'datasets/selected_data_1.xlsx'
df = pd.read_excel(excel_file)

# Assuming your features are in columns 'feature1', 'feature2', ..., 'featureN'
# and your target variable is in column 'target'
X = df.drop(columns=['Completed'])  # Features
y = df['Completed']  # Target variable

# Step 2: Initialize the StandardScaler
scaler = StandardScaler()

# Step 3: Fit and transform the data
X_scaled = scaler.fit_transform(X)

# Step 4: Now X_scaled contains your normalized features


# Optionally, you can save the normalized data to a new Excel file
normalized_data_df = pd.DataFrame(X_scaled, columns=X.columns)
normalized_data_df['Completed'] = y  # Add the target variable back if needed
normalized_data_df.to_excel('normalized_selected_data_2.xlsx', index=False)
