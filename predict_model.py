import socket
import numpy as np
import torch
import torch.nn as nn
import joblib
import csv
import os
import time

# TORCS server settings
SERVER_IP = 'localhost'
SERVER_PORT = 3001

MODEL_PATH = "expert_model.pt"
SCALER_PATH = "expert_scaler.save"
CURRENT_TRACK = "alpine-2"

feature_cols = [
    "SPEED", "TRACK_POSITION", "ANGLE_TO_TRACK_AXIS"
] + [f"TRACK_EDGE_{i}" for i in range(19)]

# PyTorch modeldefinitie (moet gelijk zijn aan train script)
class TorcsNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# Laad scaler en model
scaler = joblib.load(SCALER_PATH)
input_dim = len(feature_cols)
model = TorcsNet(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Maak een UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Eerste handshake (TORCS verwacht een init-bericht)
init_msg = b'SCR\n'
sock.sendto(init_msg, (SERVER_IP, SERVER_PORT))

print("Verbonden met TORCS op poort 3001. Wachten op sensordata...")

current_time = time.strftime("%Y%m%d_%H%M%S")
LOG_PATH = f"log_{CURRENT_TRACK}_{current_time}.csv"

# Maak logbestand aan en schrijf header (overschrijft bij elke run)
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "speed", "track_pos", "angle", "rpm", "gear", "acceleration", "brake", "steering", "new_gear"
    ])

prev_steering = 0.0  # Voor low-pass filter
alpha = 1.0  # mate van demping, 0.0 = geen demping, 1.0 = alleen nieuwe waarde

while True:
    data, addr = sock.recvfrom(1024)
    msg = data.decode()
    if not msg.strip().startswith('('):
        continue

    # Parse sensordata
    def get_value(name, default=0.0):
        import re
        match = re.search(r'\({}\s+([^)]+)\)'.format(name), msg)
        if match and len(match.groups()) > 0:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return default
        return default

    speed = get_value('speedX')
    track_pos = get_value('trackPos')
    angle = get_value('angle')
    rpm = get_value('rpm')
    gear = int(get_value('gear', 1))
    last_lap_time = get_value('lastLapTime', 0.0)

    # Track sensors (19 waardes)
    import re
    track_match = re.search(r'(track\s+([^)]+))', msg)
    if track_match:
        track = [float(x) for x in track_match.group(2).split()]
    else:
        track = [0.0]*19

    # Maak feature dict
    feature_dict = {
        "SPEED": speed,
        "TRACK_POSITION": track_pos,
        "ANGLE_TO_TRACK_AXIS": angle,
    }
    for i in range(19):
        feature_dict[f"TRACK_EDGE_{i}"] = track[i] if i < len(track) else 0.0

    # Maak feature vector
    X = np.array([[feature_dict[col] for col in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    # pick device
    with torch.no_grad():
        action = model(X_tensor).cpu().numpy()[0]
    acceleration = float(np.clip(action[0], 0.4, 0.8))
    brake = float(np.clip(action[1], 0, 0))
    steering = float(action[2])

    # Low-pass filter op stuurinput
    steering = alpha * steering + (1 - alpha) * prev_steering
    prev_steering = steering

    # Simpele schakellogica
    new_gear = gear
    if rpm > 8000:
        new_gear += 1
    elif rpm < 5000 and gear > 1:
        new_gear -= 1

    # Log naar CSV (altijd uitvoeren, vóór continue/return!)
    import time
    print(f"LOG: {speed=}, {track_pos=}, {angle=}, {rpm=}, {gear=}, {acceleration=}, {brake=}, {steering=}, {new_gear=}")
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(), speed, track_pos, angle, rpm, gear, acceleration, brake, steering, new_gear
        ])

    # Bouw actie-string voor TORCS
    action_str = f"(accel {acceleration}) (brake {brake}) (steer {steering}) (gear {new_gear})\n"
    print("Actie naar TORCS:", action_str.strip())
    sock.sendto(action_str.encode(), (SERVER_IP, SERVER_PORT))
