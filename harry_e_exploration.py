import pandas as pd
providers = pd.read_csv("2026-ASA-DataFest-Data-Files/providers.csv")
departments = pd.read_csv("2026-ASA-DataFest-Data-Files/departments.csv")
diagnosis = pd.read_csv("2026-ASA-DataFest-Data-Files/diagnosis.csv")
encounters = pd.read_csv("2026-ASA-DataFest-Data-Files/encounters.csv")
patients = pd.read_csv("2026-ASA-DataFest-Data-Files/patients.csv")
social_determinants = pd.read_csv("2026-ASA-DataFest-Data-Files/social_determinants.csv", dtype={4: str})
tigercensuscodes = pd.read_csv("2026-ASA-DataFest-Data-Files/tigercensuscodes.csv")
tornado = pd.read_csv("dataset.csv")

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



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ── STAGE 2: AGGREGATE TO MONTHLY ───────────────────────────────────────────

data['Date'] = pd.to_datetime(data['Date'])
data['YearMonth'] = data['Date'].dt.to_period('M')

# Count of encounters per city per diagnosis group per month
monthly = (data.groupby(['YearMonth', 'City', 'GroupName'])
               .size()
               .reset_index(name='Count'))

# Most common diagnosis group per city per month (type target)
dominant_type = (data.groupby(['YearMonth', 'City'])['GroupName']
                     .agg(lambda x: x.value_counts().index[0])
                     .reset_index(name='DominantType'))

monthly = monthly.merge(dominant_type, on=['YearMonth', 'City'], how='left')

print(monthly.shape)
print(monthly.head(10))

# ── STAGE 3: FEATURE ENGINEERING ────────────────────────────────────────────

monthly = monthly.sort_values(['City', 'GroupName', 'YearMonth']).reset_index(drop=True)

# Lag features
for lag in [1, 2, 3]:
    monthly[f'count_lag_{lag}'] = (monthly.groupby(['City', 'GroupName'])['Count']
                                           .shift(lag))

# Rolling 3 month mean
monthly['rolling_mean_3m'] = (monthly.groupby(['City', 'GroupName'])['Count']
                                      .shift(1)
                                      .groupby([monthly['City'], monthly['GroupName']])
                                      .transform(lambda x: x.rolling(3).mean()))

# Temporal features
monthly['month'] = monthly['YearMonth'].dt.month
monthly['season'] = monthly['month'].map({
    12: 0, 1: 0, 2: 0,
    3: 1,  4: 1, 5: 1,
    6: 2,  7: 2, 8: 2,
    9: 3,  10: 3, 11: 3
})

# Encode categoricals
city_encoder  = LabelEncoder()
group_encoder = LabelEncoder()
type_encoder  = LabelEncoder()

monthly['city_enc']  = city_encoder.fit_transform(monthly['City'])
monthly['group_enc'] = group_encoder.fit_transform(monthly['GroupName'])
monthly['type_enc']  = type_encoder.fit_transform(monthly['DominantType'])

# Drop rows with NaN from lag window
monthly = monthly.dropna()
print(f"After dropping NaN: {monthly.shape}")

# ── STAGE 4: TRAIN / TEST SPLIT ─────────────────────────────────────────────

feature_cols = [
    'count_lag_1', 'count_lag_2', 'count_lag_3',
    'rolling_mean_3m', 'month', 'season',
    'city_enc', 'group_enc'
]

train = monthly[monthly['YearMonth'].dt.year < 2025]
test  = monthly[monthly['YearMonth'].dt.year == 2025]

X_train = train[feature_cols].values.astype(np.float32)
X_test  = test[feature_cols].values.astype(np.float32)

y_count_train = train['Count'].values.astype(np.float32)
y_count_test  = test['Count'].values.astype(np.float32)

y_type_train  = train['type_enc'].values.astype(np.int64)
y_type_test   = test['type_enc'].values.astype(np.int64)

num_classes = len(type_encoder.classes_)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {num_classes}")

# ── STAGE 5: MODEL ───────────────────────────────────────────────────────────

class MultiOutputFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        self.count_head = nn.Sequential(
            nn.Linear(prev_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(prev_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        shared = self.backbone(x)
        return self.count_head(shared).squeeze(1), self.type_head(shared)


# ── TRAINING ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_count_train),
    torch.tensor(y_type_train)
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MultiOutputFNN(
    input_dim=len(feature_cols),
    hidden_dims=[256, 128, 64],
    num_classes=num_classes
).to(device)

count_criterion = nn.MSELoss()
type_criterion  = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
alpha = 0.5  # balance between count and type loss

for epoch in range(50):
    model.train()
    total_loss = 0

    for X_batch, y_count_batch, y_type_batch in loader:
        X_batch       = X_batch.to(device)
        y_count_batch = y_count_batch.to(device)
        y_type_batch  = y_type_batch.to(device)

        optimiser.zero_grad()
        count_pred, type_pred = model(X_batch)

        loss = alpha * count_criterion(count_pred, y_count_batch) + \
               (1 - alpha) * type_criterion(type_pred, y_type_batch)

        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")

# ── EVALUATION ───────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    count_pred, type_pred = model(torch.tensor(X_test).to(device))
    count_pred_np = count_pred.cpu().numpy()
    type_pred_np  = type_pred.cpu().argmax(dim=1).numpy()

print(f"\nTest MAE  (count): {mean_absolute_error(y_count_test, count_pred_np):.2f}")
print(f"Test Acc  (type):  {accuracy_score(y_type_test, type_pred_np):.3f}")
print(f"Test F1   (type):  {f1_score(y_type_test, type_pred_np, average='weighted'):.3f}")

results = pd.DataFrame({
    'City':           test['City'].values,
    'GroupName':      test['GroupName'].values,
    'YearMonth':      test['YearMonth'].astype(str).values,
    'Actual_Count':   y_count_test,
    'Predicted_Count': count_pred_np.round().astype(int),
    'Actual_Type':    type_encoder.inverse_transform(y_type_test),
    'Predicted_Type': type_encoder.inverse_transform(type_pred_np)
})

print("\n── Sample Predictions ─────────────────────────")
print(results.head(20).to_string(index=False))






tornado['BEGIN_DATE'] = pd.to_datetime(tornado['BEGIN_DATE'])
tornado['YearMonth'] = tornado['BEGIN_DATE'].dt.to_period('M')
tornado['City'] = tornado['City'].str.lower()

# Aggregate tornado events per city per month
tornado_monthly = tornado.groupby(['City', 'YearMonth']).agg(
    tornado_count=('EVENT_ID', 'count'),
    avg_distance_km=('distance_km', 'mean'),
    max_f_scale=('TOR_F_SCALE', 'max'),
    direct_injuries=('INJURIES_DIRECT', 'sum')
).reset_index()

# Merge onto your existing monthly dataframe
monthly_with_tornado = monthly.merge(tornado_monthly, on=['City', 'YearMonth'], how='left')
monthly_with_tornado[['tornado_count', 'avg_distance_km', 'max_f_scale', 'direct_injuries']] = \
    monthly_with_tornado[['tornado_count', 'avg_distance_km', 'max_f_scale', 'direct_injuries']].fillna(0)

# Flag months with a nearby tornado (within 50km)
monthly_with_tornado['tornado_nearby'] = (
    (monthly_with_tornado['tornado_count'] > 0) & 
    (monthly_with_tornado['avg_distance_km'] < 50)
).astype(int)

print(monthly_with_tornado[['City', 'YearMonth', 'Count', 'tornado_count', 'avg_distance_km', 'tornado_nearby']].head(20))
print(f"\nMonths with nearby tornado: {monthly_with_tornado['tornado_nearby'].sum()}")