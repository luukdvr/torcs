import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib
import csv

# Bestandspad
csv_file = "dataset.csv"

# Probeer delimiter automatisch te bepalen
try:
    with open(csv_file, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        delimiter = csv.Sniffer().sniff(sample).delimiter
except Exception:
    delimiter = ','  # fallback

# Inlezen CSV
data = pd.read_csv(csv_file, delimiter=delimiter)

# Strip spaties uit kolomnamen
data.columns = data.columns.str.strip().str.lower()

# Corrigeer eventuele naamfouten
data = data.rename(columns={
    'track position': 'track_position',
    'angle to track axis': 'angle_to_track_axis',
    'acceleration': 'acceleration',
    'brake': 'brake',
    'steering': 'steering'
})

# Selecteer input features en target outputs
input_features = ['speed', 'track_position', 'angle_to_track_axis']
input_features += [col for col in data.columns if "track edge sensor" in col]

target_outputs = ['steering', 'acceleration', 'brake']

# Controle
print("Gebruikte input features:", input_features)
print("Gebruikte target outputs:", target_outputs)

# Verwijder niet-numerieke tekens
data = data.replace(r"[^\d\.\-\+eE]", "", regex=True)
data = data.dropna()

# Zet kolommen naar floats
data[input_features + target_outputs] = data[input_features + target_outputs].astype(float)

# Features en labels
X = data[input_features]
y = data[target_outputs]

# Schalen van input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Bouw het neuraal netwerk
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='linear')  # Outputs: steering, acceleration, brake
])

# Compileer het model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train het model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluatie op testset
loss, mae = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, MAE: {mae:.4f}")

# Opslaan model en scaler
model.save("torcs_driver_model.h5")
joblib.dump(scaler, "scaler.save")

print("✅ Model en scaler opgeslagen.")
