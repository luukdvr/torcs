import pandas as pd
import numpy as np
import glob

def analyze_race_data():
    csv_file = "test_data/log_alpine-1_20250629_202335.csv"    
    df = pd.read_csv(csv_file)
    
    results = []
    
    for lap_num in df['lap'].unique():
        lap_data = df[df['lap'] == lap_num]
        
        if len(lap_data) > 1:
            avg_speed_kmh = lap_data['speed'].mean()
            
            off_track_count = len(lap_data[abs(lap_data['track_pos']) > 1.0])
            
            avg_steering = lap_data['steering'].mean()
            
            lap_time = lap_data['current_lap_time'].max()
            
            results.append({
                'lap': int(lap_num),
                'Gem. snelheid (km/u)': round(avg_speed_kmh, 2),
                'Off-track (x)': off_track_count,
                'Gem. stuurhoek': round(avg_steering, 4),
                'Rondetijd (s)': round(lap_time, 2)
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('lap')
    
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    analyze_race_data()