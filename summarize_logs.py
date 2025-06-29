import pandas as pd
import os
import numpy as np

LOG_PATH = "log.csv"
SUMMARY_PATH = "summary.txt"

# Lees log.csv in
if not os.path.exists(LOG_PATH):
    print(f"{LOG_PATH} niet gevonden.")
    exit(1)

df = pd.read_csv(LOG_PATH)
if df.empty:
    print("Logbestand is leeg.")
    exit(1)

# Gebruik alleen de eerste ronde (alles uit log)
mean_speed = df['speed'].mean()
off_track_count = df['off_track'].sum() if 'off_track' in df.columns else 0
mean_steering = df['abs_steering'].mean() if 'abs_steering' in df.columns else df['steering'].abs().mean()
lap_time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0] if len(df) > 1 else 0

header = f"| Race | Gem. snelheid (km/u) | Off-track (x) | Gem. stuurhoek | Rondetijd (s) |\n"
sep =    f"|------|----------------------|---------------|----------------|---------------|\n"
rows = ""

# Eerste ronde: echte data
rows += f"|    1 | {mean_speed:>20.1f} | {int(off_track_count):>13} | {mean_steering:>14.2f} | {lap_time:>13.1f} |\n"

with open(SUMMARY_PATH, "w") as f:
    f.write(header)
    f.write(sep)
    f.write(rows)

print(f"Samenvatting van 10 races/rondes opgeslagen in {SUMMARY_PATH}")