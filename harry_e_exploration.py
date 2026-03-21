import pandas as pd
providers = pd.read_csv("2026-ASA-DataFest-Data-Files/providers.csv")
departments = pd.read_csv("2026-ASA-DataFest-Data-Files/departments.csv")
diagnosis = pd.read_csv("2026-ASA-DataFest-Data-Files/diagnosis.csv")
encounters = pd.read_csv("2026-ASA-DataFest-Data-Files/encounters.csv")
patients = pd.read_csv("2026-ASA-DataFest-Data-Files/patients.csv")
social_determinants = pd.read_csv("2026-ASA-DataFest-Data-Files/social_determinants.csv", dtype={4: str})
tigercensuscodes = pd.read_csv("2026-ASA-DataFest-Data-Files/tigercensuscodes.csv")
tornados = pd.read_csv("dataset.csv")

#print(providers.columns)
#print(departments.columns)
#print(diagnosis.columns)
#print(encounters.columns)
#print(patients.columns)
#print(social_determinants.columns)
#print(tigercensuscodes.columns)

# prediction model will be based off following variables
# encounters - date, department key, primary diagnosis key
# diagnosis - diagnosis, group code
# departments - department key, city

encounters_filtered = encounters[['Date', 'DepartmentKey', 'PrimaryDiagnosisKey']]

data = encounters_filtered.merge(departments[['DepartmentKey', 'City']], on='DepartmentKey', how='left')
data = data.merge(diagnosis[['DiagnosisKey', 'GroupName', 'GroupCode']], left_on='PrimaryDiagnosisKey', right_on='DiagnosisKey', how='left')
data = data.drop(columns='DiagnosisKey')
data = data.sort_values('Date').reset_index(drop=True)
data['City'] = data['City'].str.lower()

#print(data.head(20))
#print(data['City'].unique())
#print(data['Date'].iloc[-1])

#print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
#print(data['GroupName'].value_counts().head(10))
#print(data['City'].value_counts().head(10))

data = data.dropna()
#print(data.shape)