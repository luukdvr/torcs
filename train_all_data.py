import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Pad naar de map met trainingsdata
DATA_DIR = os.path.join('train_data', 'train_data')
EXPECTED_COLS = 24  # Aantal verwachte kolommen (sensoren/features)

# Definieer de gewenste kolomnamen (voorbeeld van f-speedway.csv)
expected_columns = [
    'ACCELERATION', 'BRAKE', 'STEERING', 'SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS',
    'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4',
    'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9',
    'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14',
    'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17'
]

# Normaliseer expected_columns voor vergelijking
expected_set = set([col.strip().upper() for col in expected_columns])

# --- Fix: combined_data.csv opnieuw aanmaken als kolommen niet kloppen ---
combined_path = os.path.join(DATA_DIR, 'combined_data.csv')
source_path = os.path.join(DATA_DIR, 'f-speedway.csv')
if os.path.exists(combined_path):
    df_check = pd.read_csv(combined_path, nrows=1)
    if not all([col in df_check.columns for col in expected_columns]):
        print('[INFO] combined_data.csv heeft geen correcte kolomnamen. Maak opnieuw aan uit f-speedway.csv.')
        df_src = pd.read_csv(source_path)
        # Kolommen in juiste volgorde en naam
        col_map = {col.strip().upper(): col for col in df_src.columns}
        df_ordered = df_src[[col_map[col] for col in [c.strip().upper() for c in expected_columns]]]
        df_ordered.columns = expected_columns
        df_ordered.to_csv(combined_path, index=False)
        print('[INFO] combined_data.csv opnieuw aangemaakt.')

# Alleen combined_data.csv gebruiken voor training
fpath = os.path.join(DATA_DIR, 'combined_data.csv')
try:
    df = pd.read_csv(fpath)
    orig_cols = list(df.columns)
    norm_cols = [col.strip().upper() for col in orig_cols]
    print(f"\nBestand: combined_data.csv")
    print(f"Aantal kolommen: {df.shape[1]}")
    print(f"Kolomnamen: {orig_cols}")
    print(f"Eerste 3 rijen:\n{df.head(3)}")
    # Controleer of alle expected_columns aanwezig zijn
    missing = [col for col in expected_columns if col not in orig_cols and col not in norm_cols]
    if missing:
        raise ValueError(f"combined_data.csv mist de volgende kolommen: {missing}")
    # Kolommen in juiste volgorde zetten
    col_map = {col.strip().upper(): col for col in orig_cols}
    df_ordered = df[[col_map[col] for col in [c.strip().upper() for c in expected_columns]]]
    print(f"Toegevoegd voor training: {df_ordered.shape[0]} rijen.")
    filtered_dataframes = [df_ordered]
except Exception as e:
    raise ValueError(f"Fout bij lezen of verwerken van combined_data.csv: {e}")

if not filtered_dataframes:
    raise ValueError("Geen geldige data gevonden!")

# Combineer alle geldige dataframes
all_data = pd.concat(filtered_dataframes, ignore_index=True)
print(f"Totale rijen na samenvoegen: {all_data.shape[0]}")
# Verwijder rijen met NaN na inlezen
all_data = all_data.dropna()
print(f"Totale rijen na verwijderen van NaN: {all_data.shape[0]}")

# Optioneel: sla gecombineerde data op
all_data.to_csv('combined_data.csv', index=False)

# Splits features en targets (eerste 3 kolommen = targets, rest = features)
target_cols = ['ACCELERATION', 'BRAKE', 'STEERING']
feature_cols = [col for col in all_data.columns if col not in target_cols]
X = all_data[feature_cols]
y = all_data[target_cols]

# Controleer op NaN, inf, -inf in features en targets
print('Eerste 5 rijen van features (X):')
print(X.head())
print('Eerste 5 rijen van targets (y):')
print(y.head())

if X.isnull().values.any() or y.isnull().values.any():
    raise ValueError('Er zitten NaN waarden in de features of targets!')
if np.isinf(X.values).any() or np.isinf(y.values).any():
    raise ValueError('Er zitten inf of -inf waarden in de features of targets!')

print('Geen NaN of inf gevonden in features of targets.')

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Schalen (fit op DataFrame met kolomnamen)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sla scaler op
joblib.dump(scaler, 'scaler.save')

# Simpel Keras model (pas aan indien gewenst)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# Debug-statistieken voor targets
print('\n[DEBUG] Target-statistieken:')
for col in target_cols:
    print(f"{col}: min={y[col].min():.4f}, max={y[col].max():.4f}, mean={y[col].mean():.4f}, std={y[col].std():.4f}")

# --- Extra check: toon shape en eerste rijen van combined_data.csv ---
print(f"[CHECK] combined_data.csv shape: {all_data.shape}")
print(f"[CHECK] Eerste 3 rijen:\n{all_data.head(3)}")

# Train model
model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_test_scaled, y_test))

# Sla model op
model.save('torcs_driver_model.h5')
print("Model en scaler opgeslagen!")
