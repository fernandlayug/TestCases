import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the input and output file paths
input_excel_file = "datasets/selected_data_1.xlsx"
output_excel_file = "datasets/normalized_selected_data_1.xlsx"

# Read data from Excel file
data = pd.read_excel(input_excel_file)

# Separate features and target variable (assuming last column is target)
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize numerical columns
numerical_columns = features.select_dtypes(include=['number']).columns
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

# Save the normalized data to an Excel file
normalized_data = pd.concat([features, target], axis=1)
normalized_data.to_excel(output_excel_file, index=False)

print(f"Normalized data saved to {output_excel_file}")
