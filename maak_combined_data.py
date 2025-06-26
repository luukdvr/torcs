import os
import pandas as pd
import numpy as np

# Pad naar de map met alle trainingsdata
DATA_DIR = os.path.join('train_data', 'train_data')
# Verwachte kolommen in de juiste volgorde
expected_columns = [
    'ACCELERATION', 'BRAKE', 'STEERING', 'SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS',
    'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4',
    'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9',
    'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14',
    'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17'
]

all_dfs = []
for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.csv'):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    try:
        df = pd.read_csv(fpath, header=0)
        orig_cols = [col.strip().upper() for col in df.columns]
        if not all(col in orig_cols for col in [c.strip().upper() for c in expected_columns]):
            # Probeer opnieuw in te lezen zonder header
            df = pd.read_csv(fpath, header=None)
            # Als 25 kolommen: neem alleen de eerste 24 (laatste kolom is dummy/extra)
            if df.shape[1] == len(expected_columns) + 1:
                df = df.iloc[:, :len(expected_columns)]
                df.columns = expected_columns
                print(f"[WAARSCHUWING] {fname} had 25 kolommen zonder header, eerste 24 gebruikt.")
            elif df.shape[1] == len(expected_columns):
                df.columns = expected_columns
                print(f"[WAARSCHUWING] {fname} had geen header, expected_columns als kolomnamen gebruikt.")
            else:
                print(f"Overgeslagen (verkeerd aantal kolommen): {fname}")
                continue
        else:
            # Kolommen in juiste volgorde zetten
            col_map = {col.strip().upper(): col for col in df.columns}
            df = df[[col_map[c.strip().upper()] for c in expected_columns]]
            df.columns = expected_columns
        all_dfs.append(df)
        print(f"Toegevoegd: {fname} ({df.shape[0]} rijen)")
    except Exception as e:
        print(f"Fout bij verwerken {fname}: {e}")

if not all_dfs:
    raise ValueError("Geen geldige CSV's gevonden!")

combined = pd.concat(all_dfs, ignore_index=True)
combined = combined.dropna()
combined.to_csv(os.path.join(DATA_DIR, 'combined_data.csv'), index=False)
print(f"Samengevoegd bestand opgeslagen als combined_data.csv met {combined.shape[0]} rijen.")
