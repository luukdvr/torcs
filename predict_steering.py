import socket
import numpy as np
from keras.models import load_model
import joblib

# TORCS server settings
SERVER_IP = 'localhost'
SERVER_PORT = 3001

# Laad model en scaler
model = load_model("steering_model.h5", compile=False)
scaler = joblib.load("steering_scaler.save")

# Maak een UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Eerste handshake (TORCS verwacht een init-bericht)
init_msg = b'SCR\n'
sock.sendto(init_msg, (SERVER_IP, SERVER_PORT))

print("Verbonden met TORCS op poort 3001. Wachten op sensordata...")

while True:
    data, addr = sock.recvfrom(1024)
    msg = data.decode()
    if not msg.strip().startswith('('):
        continue

    # Parse trackPos en angle
    import re
    def get_value(name, default=0.0):
        match = re.search(r'\({}\s+([^)]+)\)'.format(name), msg)
        if match and len(match.groups()) > 0:
            try:
                return float(match.group(2))
            except (ValueError, IndexError):
                return default
        return default

    track_pos = get_value('trackPos')
    angle = get_value('angle')

    # Maak input voor model
    input_data = np.array([[track_pos, angle]])
    input_scaled = scaler.transform(input_data)
    steering = float(np.clip(model.predict(input_scaled)[0][0], -1, 1))

    # Simpel gas geven, niet remmen
    acceleration = 0.5
    brake = 0.0

    action = f"(accel {acceleration}) (brake {brake}) (steer {steering})\n"
    print(f"trackPos={track_pos:.3f}, angle={angle:.3f} => steer={steering:.3f}")
    sock.sendto(action.encode(), (SERVER_IP, SERVER_PORT))
