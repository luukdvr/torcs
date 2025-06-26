import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Laad de data
csv_file = "dataset.csv"
data = pd.read_csv(csv_file)

# Strip spaties uit kolomnamen en zet alles lowercase
cols = [c.strip().lower() for c in data.columns]
data.columns = cols

# Corrigeer kolomnamen
rename = {
    'track position': 'track_position',
    'angle to track axis': 'angle_to_track_axis',
    'steering': 'steering',
    'acceleration': 'acceleration',
    'brake': 'brake'
}
data = data.rename(columns=rename)

# Selecteer alleen de relevante kolommen
input_features = ['track_position', 'angle_to_track_axis']
target_output = ['steering']

# Verwijder niet-numerieke tekens en drop NaN
for col in input_features + target_output:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=input_features + target_output)

# Features en labels
X = data[input_features]
y = data[target_output]

# Schalen
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(len(input_features),)),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh')  # stuurhoek tussen -1 en 1
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Test
loss, mae = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, MAE: {mae:.4f}")

# Opslaan
model.save("steering_model.h5")
joblib.dump(scaler, "steering_scaler.save")
print("✅ Model en scaler opgeslagen.")
