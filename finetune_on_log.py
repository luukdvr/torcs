import pandas as pd
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping
import joblib
from sklearn.preprocessing import StandardScaler
import os
import csv

# Bestandsnamen
LOGFILE = "driving_log.csv"
MODELFILE = "torcs_driver_model.h5"
SCALERFILE = "scaler.save"

# Laad logdata
assert os.path.isfile(LOGFILE), f"Logbestand {LOGFILE} niet gevonden. Rijd eerst een sessie."
df = pd.read_csv(LOGFILE)

# Mapping van log-kolommen naar scaler/model feature-namen
log_to_model = {
    "speed_x": "SPEED",
    "track_pos": "TRACK_POSITION",
    "angle": "ANGLE_TO_TRACK_AXIS",
}
for i in range(19):
    log_to_model[f"track_{i}"] = f"TRACK_EDGE_{i}"

# Hernoem kolommen in df
renamed = df.rename(columns=log_to_model)

# Laad scaler en bepaal feature volgorde
scaler = joblib.load(SCALERFILE)
feature_cols = list(scaler.feature_names_in_)

# Targets
target_cols = ["acceleration", "brake", "steering"]

# Gebruik alleen de juiste features
X = renamed[feature_cols].values
Y = renamed[target_cols].values

# Transformeer features
X_scaled = scaler.transform(X)

# Laad bestaand model
assert os.path.isfile(MODELFILE), f"Modelbestand {MODELFILE} niet gevonden."
model = load_model(MODELFILE, compile=False)
model.compile(optimizer="adam", loss="mse")

# Fine-tune model op nieuwe data
print(f"Start fine-tuning op {X.shape[0]} nieuwe samples...")
history = model.fit(
    X_scaled, Y,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=2
)

# Sla verbeterd model en scaler op
model.save(MODELFILE)
joblib.dump(scaler, SCALERFILE)
print("Fine-tuning klaar. Model en scaler zijn bijgewerkt!")

# Leeg driving_log.csv behalve header
with open(LOGFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "speed_x", "track_pos", "angle", *[f"track_{i}" for i in range(19)],
        "acceleration", "brake", "steering", "gear", "reward"
    ])
print(f"{LOGFILE} is geleegd. Nieuwe data wordt vanaf nu gelogd.")
