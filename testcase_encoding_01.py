import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Read the Excel file
df = pd.read_excel('datasets/imputed_ALSdata.xlsx')

# Step 2: Encode the columns

# Ordinal Encoding for the 'EducationalAttainment' and 'DaysAvailable' columns
# Define a mapping for ordinal encoding
mothereducation_mapping = {'None': 0, 'Primary Education': 1, 'Secondary Education': 2, 'College level': 3}
df['MotherEducation'] = df['MotherEducation'].map(mothereducation_mapping)

mothersalary_mapping = {0: 0, '1-5,000': 1, '5,001-15,000': 2, '15,001-25,000': 3, '25,001-35,000': 4, '35,001-45,000': 5, '45,001 and above': 6}
df['MotherSalaryRange'] = df['MotherSalaryRange'].map(mothersalary_mapping)

fathereducation_mapping = {'None': 0, 'Primary Education': 1, 'Secondary Education': 2, 'College level': 3}
df['FatherEducation'] = df['FatherEducation'].map(fathereducation_mapping)

fathersalary_mapping = {0: 0, '1-5,000': 1, '5,001-15,000': 2, '15,001-25,000': 3, '25,001-35,000': 4, '35,001-45,000': 5, '45,001 and above': 6}
df['FatherSalaryRange'] = df['FatherSalaryRange'].map(fathersalary_mapping)

education_mapping = {'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4, 'Grade 5': 5, 'Grade 6': 6, 'Grade 7': 7, 'Grade 8': 8, 'Grade 9': 9}
df['EducationalAttainment'] = df['EducationalAttainment'].map(education_mapping)

traveltime_mapping = {'less than 15 minutes': 1, '15 minutes - 30 minutes': 2, '30 minutes - 1 hour': 3, 'more than 1 hour': 4}
df['TravelTime'] = df['TravelTime'].map(traveltime_mapping)

distance_mapping = {'1-3.9km': 1, '4-7.9km': 2, '8-10km': 3, '10.1-14.9km': 4, '15-17.9km.': 5, '18-20km': 6, '20.1kmAbove': 7}
df['DistanceHomeSchool'] = df['DistanceHomeSchool'].map(distance_mapping)

days_mapping = {'1 Day': 1, '2 Days': 2, '3 Days': 3, '4 Days': 4, '5 Days': 5}
df['DaysAvailable'] = df['DaysAvailable'].map(days_mapping)


# One-Hot Encoding for nominal columns
nominal_columns = [
    'CivilStatus', 'HomeAddressType', 'MotherJob','FatherJob','Guardian','ParentCohabitation','SchoolAttended','ReasonNotAttendingSchool','Transportation'
]

for column in nominal_columns:
    df = pd.get_dummies(df, columns=[column], prefix=column, dtype=int)

# Binary Encoding for binary columns using Label Encoding
binary_columns = [
    'Gender', 'Child', 'FinancialSupport','FamilyEducationalSupport','PursueCollege','InternetAccess','Job','FacetoFace','AttendClassRegularly','OnTimeSubmission','Completed'
]

label_encoder = LabelEncoder()
for column in binary_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Ensure 'Completed' is the last column
completed = df.pop('Completed')
df['Completed'] = completed

# Step 3: Save the encoded DataFrame to a new Excel file
df.to_excel('datasets/encoded_data_1.xlsx', index=False)

# Print the resulting DataFrame to verify the encodings
print(df)
