import socket
import numpy as np
from keras.models import load_model
import joblib
import csv
import os

# TORCS server settings
SERVER_IP = 'localhost'
SERVER_PORT = 3001

# Laad model en scaler
model = load_model("torcs_driver_model.h5", compile=False)
scaler = joblib.load("scaler.save")

# Maak een UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Eerste handshake (TORCS verwacht een init-bericht)
init_msg = b'SCR\n'
sock.sendto(init_msg, (SERVER_IP, SERVER_PORT))

print("Verbonden met TORCS op poort 3001. Wachten op sensordata...")

current_gear = 1  # Houd de huidige versnelling bij buiten de loop

logfile = "driving_log.csv"
log_exists = os.path.isfile(logfile)

with open(logfile, "a", newline="") as csvfile:
    logwriter = csv.writer(csvfile)
    if not log_exists:
        # Schrijf header
        logwriter.writerow([
            "speed_x", "track_pos", "angle", *[f"track_{i}" for i in range(19)],
            "acceleration", "brake", "steering", "gear", "reward"
        ])

while True:
    # Ontvang sensordata van TORCS
    data, addr = sock.recvfrom(1024)
    msg = data.decode()
    print("Ontvangen van TORCS:", msg)

    # Sla berichten zonder sensordata over
    if not msg.strip().startswith('('):
        continue

    # Parse sensordata (voorbeeld: haal speedX, trackPos, angle, track[] eruit)
    def get_value(name, default=0.0):
        import re
        match = re.search(r'\({}\s+([^)]+)\)'.format(name), msg)
        if match and len(match.groups()) > 0:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return default
        return default

    # Parse ook gear en rpm
    def get_int_value(name, default=0):
        import re
        match = re.search(r'\({}\s+([^)]+)\)'.format(name), msg)
        if match and len(match.groups()) > 0:
            try:
                return int(float(match.group(1)))
            except (ValueError, IndexError):
                return default
        return default

    speed_x = get_value('speedX')
    track_pos = get_value('trackPos')
    angle = get_value('angle')
    rpm = get_value('rpm')
    gear = get_int_value('gear')
    # Track sensors (19 waardes)
    import re
    track_match = re.search(r'(track\s+([^)]+))', msg)
    if track_match:
        track = [float(x) for x in track_match.group(2).split()]
    else:
        track = [0.0]*19

    # Zet sensordata om naar juiste DataFrame voor scaler
    import pandas as pd
    feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    if feature_names is not None:
        print(f"[DEBUG] Aantal features verwacht door scaler: {len(feature_names)}")
        print(f"[DEBUG] Feature-namen scaler: {feature_names}")
        feature_dict = {name: 0.0 for name in feature_names}
        # Vul features exact zoals de scaler verwacht
        if 'SPEED' in feature_dict:
            feature_dict['SPEED'] = speed_x
        if 'TRACK_POSITION' in feature_dict:
            feature_dict['TRACK_POSITION'] = track_pos
        if 'ANGLE_TO_TRACK_AXIS' in feature_dict:
            feature_dict['ANGLE_TO_TRACK_AXIS'] = angle
        for i in range(18):  # 0 t/m 17
            key = f'TRACK_EDGE_{i}'
            if key in feature_dict and i < len(track):
                feature_dict[key] = track[i]
        input_df = pd.DataFrame([[feature_dict[name] for name in feature_names]], columns=feature_names)
        print("Scaler features:", feature_names)
        print("Input DataFrame:\n", input_df)
        input_scaled = scaler.transform(input_df)
    else:
        raise ValueError("Scaler heeft geen feature_names_in_. Kan input niet correct mappen.")
    prediction = model.predict(input_scaled)[0]
    print("Model output:", prediction)
    # Let op: volgorde is [ACCELERATION, BRAKE, STEERING]
    acceleration = float(np.clip(prediction[0], 0, 1))
    brake = float(np.clip(prediction[1], 0, 1))
    steering = float(prediction[2])

    # Dynamische steer-clamp
    if speed_x > 40:
        steering = float(np.clip(steering, -0.15, 0.15))
    elif speed_x > 20:
        steering = float(np.clip(steering, -0.3, 0.3))
    else:
        steering = float(np.clip(steering, -1, 1))

    # Schakellogica
    if gear > 0:
        current_gear = gear
    if rpm > 7000 and speed_x > 10 and current_gear < 6:
        current_gear += 1
    elif rpm < 3000 and speed_x > 10 and current_gear > 1:
        current_gear -= 1
    if speed_x < 1:
        current_gear = 1

    # Reward (optioneel, simpel)
    reward = speed_x
    if abs(track_pos) > 1:
        reward -= 10

    # Bouw actie-string voor TORCS
    action = f"(accel {acceleration}) (brake {brake}) (steer {steering}) (gear {current_gear})\n"
    print("Actie naar TORCS:", action.strip())
    sock.sendto(action.encode(), (SERVER_IP, SERVER_PORT))

    # Logging
    with open(logfile, "a", newline="") as csvfile:
        logwriter = csv.writer(csvfile)
        logwriter.writerow([
            speed_x, track_pos, angle, *track,
            acceleration, brake, steering, current_gear, reward
        ])
