#!/usr/bin/env python3
"""
TORCS Data Analysis - Main Script
Supporting Chapter 6: Input Data Analysis
Autonomous TORCS Driver - Technical Computer Science Year 3

This script performs comprehensive analysis of TORCS racing telemetry data
for autonomous AI-driver development.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TORCSDataAnalyzer:
    """Main class for TORCS data analysis."""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.normalized_df = None
        self.feature_df = None
        self.scalers = {}
        
        # Define sensor groups
        self.track_edge_columns = [f'TRACK_EDGE_{i}' for i in range(19)]
        self.control_columns = ['ACCELERATION', 'BRAKE', 'STEERING']
        self.state_columns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS']
        
    def load_all_datasets(self):
        """Load all CSV files and combine into one dataset."""
        print("Loading TORCS datasets...")
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        datasets = []
        file_info = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = file.stem
                df['track_type'] = self._classify_track_type(file.stem)
                datasets.append(df)
                
                file_info.append({
                    'filename': file.name,
                    'rows': len(df),
                    'track_type': df['track_type'].iloc[0]
                })
                
                print(f"  ✓ {file.name}: {len(df):,} rows")
                
            except Exception as e:
                print(f"  ✗ Error loading {file.name}: {e}")
        
        self.df = pd.concat(datasets, ignore_index=True)
        print(f"\nTotal loaded: {len(self.df):,} rows from {len(csv_files)} files")
        
        return pd.DataFrame(file_info)
    
    def _classify_track_type(self, filename):
        """Classify track type based on filename."""
        filename = filename.lower()
        
        if any(keyword in filename for keyword in ['speedway', 'oval']):
            return 'oval'
        elif 'corner' in filename:
            return 'corners'
        elif any(keyword in filename for keyword in ['alpine', 'suzuka', 'aalborg']):
            return 'road_course'
        elif 'sprint' in filename:
            return 'sprint'
        elif 'multi' in filename:
            return 'multi_car'
        else:
            return 'mixed'
    
    def data_quality_check(self):
        """Comprehensive data quality control."""
        print("=== DATA QUALITY CHECK ===")
        
        # Basic information
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Missing values
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df) * 100).round(2)
        
        missing_report = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        })
        
        print("\nMissing Values:")
        missing_cols = missing_report[missing_report['Missing_Count'] > 0]
        if len(missing_cols) > 0:
            print(missing_cols)
        else:
            print("No missing values found!")
        
        # Data type check
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        # Range validation
        print("\n=== SENSOR VALIDATION ===")
        
        # Control inputs validation
        accel_invalid = ((self.df['ACCELERATION'] < 0) | (self.df['ACCELERATION'] > 1)).sum()
        brake_invalid = ((self.df['BRAKE'] < 0) | (self.df['BRAKE'] > 1)).sum()
        steering_invalid = ((self.df['STEERING'] < -1) | (self.df['STEERING'] > 1)).sum()
        
        print(f"Invalid ACCELERATION values: {accel_invalid}")
        print(f"Invalid BRAKE values: {brake_invalid}")
        print(f"Invalid STEERING values: {steering_invalid}")
        
        # Track position reliability
        off_track = (abs(self.df['TRACK_POSITION']) > 1).sum()
        off_track_pct = (off_track / len(self.df)) * 100
        print(f"Off-track samples: {off_track:,} ({off_track_pct:.2f}%)")
        
        # Track edge sensors validation
        track_edge_invalid = 0
        for col in self.track_edge_columns:
            if col in self.df.columns:
                invalid = ((self.df[col] < 0) | (self.df[col] > 100)).sum()
                track_edge_invalid += invalid
        
        print(f"Invalid track edge sensor values: {track_edge_invalid}")
        
        return missing_report
    
    def clean_and_validate_data(self):
        """Clean and validate data according to TORCS specifications."""
        print("Cleaning and validating data...")
        
        self.df = self.df.copy()
        
        # Clip control inputs to valid ranges
        self.df['ACCELERATION'] = self.df['ACCELERATION'].clip(0, 1)
        self.df['BRAKE'] = self.df['BRAKE'].clip(0, 1)
        self.df['STEERING'] = self.df['STEERING'].clip(-1, 1)
        
        # Clip track edge sensors to valid range
        for col in self.track_edge_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(0, 100)
        
        # Mark unreliable track edge data when off-track
        unreliable_mask = abs(self.df['TRACK_POSITION']) > 1
        available_sensors = [col for col in self.track_edge_columns if col in self.df.columns]
        self.df.loc[unreliable_mask, available_sensors] = np.nan
        
        print(f"Data cleaning completed. {unreliable_mask.sum()} track edge measurements marked as unreliable.")
    
    def generate_dataset_overview(self):
        """Generate comprehensive dataset overview."""
        print("\n=== DATASET OVERVIEW ===")
        
        # Basic metrics
        overview_stats = {
            'Total Samples': len(self.df),
            'Simulation Time (sec)': len(self.df) / 50,
            'Unique Tracks': self.df['source_file'].nunique(),
            'Track Types': self.df['track_type'].nunique(),
            'Avg Speed (m/s)': self.df['SPEED'].mean(),
            'Max Speed (m/s)': self.df['SPEED'].max(),
            'Missing Values': self.df.isnull().sum().sum(),
            'Off-track Samples': (abs(self.df['TRACK_POSITION']) > 1).sum()
        }
        
        for key, value in overview_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Track type distribution
        print(f"\nTRACK TYPE DISTRIBUTION:")
        track_dist = self.df['track_type'].value_counts()
        for track_type, count in track_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {track_type:12s}: {count:6,} samples ({percentage:5.1f}%)")
        
        return overview_stats
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        print("Calculating performance metrics...")
        
        metrics_by_track = {}
        
        for track_type in self.df['track_type'].unique():
            subset = self.df[self.df['track_type'] == track_type]
            
            metrics = {
                'sample_count': len(subset),
                'avg_speed': subset['SPEED'].mean(),
                'max_speed': subset['SPEED'].max(),
                'speed_std': subset['SPEED'].std(),
                'avg_acceleration': subset['ACCELERATION'].mean(),
                'avg_brake': subset['BRAKE'].mean(),
                'steering_consistency': subset['STEERING'].rolling(20).std().mean(),
                'track_position_std': subset['TRACK_POSITION'].std(),
                'off_track_percentage': (abs(subset['TRACK_POSITION']) > 1).mean() * 100
            }
            
            # Physics-based metrics
            subset_clean = subset.dropna()
            if len(subset_clean) > 0:
                metrics['lateral_g_avg'] = abs(subset_clean['SPEED'] * subset_clean['STEERING']).mean()
                metrics['brake_throttle_conflicts'] = ((subset_clean['ACCELERATION'] > 0.1) & 
                                                      (subset_clean['BRAKE'] > 0.1)).mean() * 100
            
            metrics_by_track[track_type] = metrics
        
        return pd.DataFrame(metrics_by_track).T
    
    def perform_correlation_analysis(self):
        """Perform comprehensive correlation analysis."""
        print("Performing correlation analysis...")
        
        # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Focus on speed correlations (key performance indicator)
        speed_correlations = correlation_matrix['SPEED'].abs().sort_values(ascending=False)
        
        print("=== TOP 10 CORRELATIONS WITH SPEED ===")
        for i, (feature, correlation) in enumerate(speed_correlations.head(11).items()):
            if feature != 'SPEED':
                print(f"{i:2d}. {feature:20s}: {correlation:.4f}")
        
        return correlation_matrix, speed_correlations
    
    def analyze_racing_lines(self):
        """Analyze racing line optimization."""
        print("\n=== RACING LINE ANALYSIS ===")
        
        # Optimal track position per speed range
        speed_bins = pd.cut(self.df['SPEED'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        optimal_positions = self.df.groupby(speed_bins, observed=True)['TRACK_POSITION'].mean()
        
        print("Optimal track position per speed range:")
        for speed_range, position in optimal_positions.items():
            print(f"  {speed_range}: {position:.3f}")
        
        # Corner entry/exit analysis
        high_steering = self.df[abs(self.df['STEERING']) > 0.3]
        corner_performance = {
            'corner_samples': len(high_steering),
            'corner_percentage': len(high_steering) / len(self.df) * 100,
            'avg_entry_speed': high_steering['SPEED'].mean(),
            'avg_track_position': high_steering['TRACK_POSITION'].mean(),
            'steering_smoothness': high_steering['STEERING'].rolling(10).std().mean()
        }
        
        print(f"\nCorner performance:")
        print(f"  Corner samples: {corner_performance['corner_samples']:,} ({corner_performance['corner_percentage']:.1f}%)")
        print(f"  Average corner speed: {corner_performance['avg_entry_speed']:.2f} m/s")
        print(f"  Average track position in corners: {corner_performance['avg_track_position']:.3f}")
        print(f"  Steering smoothness: {corner_performance['steering_smoothness']:.4f}")
        
        return optimal_positions, corner_performance
    
    def create_comprehensive_visualizations(self, save_plots=True):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('TORCS Data Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')
        
        # 1. Speed distribution by track type
        plt.subplot(3, 4, 1)
        for track_type in self.df['track_type'].unique():
            subset = self.df[self.df['track_type'] == track_type]
            plt.hist(subset['SPEED'], alpha=0.6, label=track_type, bins=30, density=True)
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')
        plt.title('Speed Distribution by Track Type')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 2. Speed vs Steering relationship
        plt.subplot(3, 4, 2)
        sample_size = min(10000, len(self.df))
        sample_data = self.df.sample(n=sample_size)
        scatter = plt.scatter(sample_data['SPEED'], abs(sample_data['STEERING']), 
                            alpha=0.4, s=1, c=sample_data['TRACK_POSITION'], cmap='RdYlBu')
        plt.colorbar(scatter, label='Track Position')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Absolute Steering Angle')
        plt.title('Speed vs Steering (color = track pos)')
        plt.grid(True, alpha=0.3)
        
        # 3. Control input correlation heatmap
        plt.subplot(3, 4, 3)
        control_corr = self.df[self.control_columns + ['SPEED']].corr()
        sns.heatmap(control_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Control Input Correlations')
        
        # 4. Track edge sensor distribution
        plt.subplot(3, 4, 4)
        available_sensors = [col for col in self.track_edge_columns if col in self.df.columns]
        sensor_means = self.df[available_sensors].mean()
        sensor_angles = np.linspace(-90, 90, len(available_sensors))
        plt.plot(sensor_angles, sensor_means, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Sensor Angle (degrees)')
        plt.ylabel('Avg Distance (m)')
        plt.title('Track Edge Sensor Profile')
        plt.grid(True, alpha=0.3)
        
        # 5. Performance by track type
        plt.subplot(3, 4, 5)
        perf_metrics = self.calculate_performance_metrics()
        perf_metrics['avg_speed'].plot(kind='bar', color='skyblue', edgecolor='black')
        plt.ylabel('Avg Speed (m/s)')
        plt.title('Performance by Track Type')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Speed time series example
        plt.subplot(3, 4, 6)
        sample_length = min(2000, len(self.df))
        time_axis = np.arange(sample_length) / 50
        plt.plot(time_axis, self.df['SPEED'].iloc[:sample_length], label='Speed', linewidth=1)
        plt.plot(time_axis, self.df['ACCELERATION'].iloc[:sample_length] * 50, 
                label='Throttle × 50', alpha=0.7, linewidth=1)
        plt.plot(time_axis, self.df['BRAKE'].iloc[:sample_length] * 50, 
                label='Brake × 50', alpha=0.7, linewidth=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Value')
        plt.title('Time Series Example (40s)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 7. Track position vs speed heatmap
        plt.subplot(3, 4, 7)
        plt.hist2d(self.df['TRACK_POSITION'], self.df['SPEED'], bins=50, cmap='viridis')
        plt.colorbar(label='Sample Count')
        plt.xlabel('Track Position')
        plt.ylabel('Speed (m/s)')
        plt.title('Track Position vs Speed')
        
        # 8. Outlier detection boxplots
        plt.subplot(3, 4, 8)
        outlier_data = [self.df['SPEED'], self.df['TRACK_POSITION'], 
                       abs(self.df['STEERING']), self.df['ACCELERATION']]
        plt.boxplot(outlier_data, labels=['Speed', 'Track\nPos', 'Abs\nSteering', 'Accel'])
        plt.title('Outlier Detection')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        
        # 9. Sensor fusion visualization
        plt.subplot(3, 4, 9)
        if all(col in self.df.columns for col in ['TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10']):
            forward_sensors = self.df[['TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10']].mean(axis=1)
            sample_data = self.df.sample(n=min(5000, len(self.df)))
            sample_forward = forward_sensors.loc[sample_data.index]
            plt.scatter(sample_forward, sample_data['SPEED'], alpha=0.4, s=1)
            plt.xlabel('Forward Clearance (m)')
            plt.ylabel('Speed (m/s)')
            plt.title('Sensor Fusion: Forward Vision')
            plt.grid(True, alpha=0.3)
        
        # 10. Racing line analysis
        plt.subplot(3, 4, 10)
        speed_bins = pd.cut(self.df['SPEED'], bins=5, labels=['Very\nLow', 'Low', 'Med', 'High', 'Very\nHigh'])
        optimal_positions = self.df.groupby(speed_bins, observed=True)['TRACK_POSITION'].mean()
        optimal_positions.plot(kind='bar', color='orange', edgecolor='black')
        plt.ylabel('Avg Track Position')
        plt.title('Optimal Racing Line')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 11. Feature importance placeholder
        plt.subplot(3, 4, 11)
        plt.text(0.5, 0.5, 'Feature Importance\n(Run feature analysis)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
        
        # 12. Data quality summary
        plt.subplot(3, 4, 12)
        quality_metrics = {
            'Valid\nSamples': len(self.df) / 1000,  # in thousands
            'Off-track\n%': (abs(self.df['TRACK_POSITION']) > 1).mean() * 100,
            'Missing\n%': self.df.isnull().mean().mean() * 100,
            'Track\nTypes': self.df['track_type'].nunique()
        }
        
        bars = plt.bar(range(len(quality_metrics)), list(quality_metrics.values()), 
                      color=['green', 'orange', 'red', 'blue'])
        plt.xticks(range(len(quality_metrics)), list(quality_metrics.keys()))
        plt.title('Data Quality Metrics')
        plt.ylabel('Count / Percentage')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('torcs_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualizations saved as 'torcs_comprehensive_analysis.png'")
        
        plt.show()
    
    def normalize_features(self):
        """Normalize data for ML training."""
        print("Normalizing data for ML training...")
        
        self.normalized_df = self.df.copy()
        
        # Min-Max scaling for bounded controls (0-1 range)
        minmax_features = ['ACCELERATION', 'BRAKE']
        self.scalers['minmax'] = MinMaxScaler()
        self.normalized_df[minmax_features] = self.scalers['minmax'].fit_transform(
            self.df[minmax_features])
        
        # Standard scaling for normally distributed variables
        standard_features = ['SPEED', 'ANGLE_TO_TRACK_AXIS']
        self.scalers['standard'] = StandardScaler()
        self.normalized_df[standard_features] = self.scalers['standard'].fit_transform(
            self.df[standard_features])
        
        # Robust scaling for outlier-sensitive sensors
        available_sensors = [col for col in self.track_edge_columns if col in self.df.columns]
        if available_sensors:
            self.scalers['robust'] = RobustScaler()
            self.normalized_df[available_sensors] = self.scalers['robust'].fit_transform(
                self.df[available_sensors])
        
        # Circular normalization for steering
        self.normalized_df['STEERING_SIN'] = np.sin(self.df['STEERING'])
        self.normalized_df['STEERING_COS'] = np.cos(self.df['STEERING'])
        
        print("Data normalization completed.")
        return self.normalized_df
    
    def engineer_racing_features(self):
        """Engineer domain-specific features."""
        print("Engineering racing-specific features...")
        
        self.feature_df = self.df.copy()
        
        # Physics-based features
        self.feature_df['lateral_acceleration'] = self.df['SPEED'] * abs(self.df['STEERING'])
        self.feature_df['deceleration_rate'] = self.df['BRAKE'] * self.df['SPEED']
        self.feature_df['acceleration_rate'] = self.df['ACCELERATION'] * (1 - self.df['BRAKE'])
        
        # Sensor fusion features
        available_sensors = [col for col in self.track_edge_columns if col in self.df.columns]
        if available_sensors:
            self.feature_df['min_distance'] = self.df[available_sensors].min(axis=1)
            
            # Forward clearance (if sensors available)
            forward_sensors = [f'TRACK_EDGE_{i}' for i in [8, 9, 10] if f'TRACK_EDGE_{i}' in self.df.columns]
            if forward_sensors:
                self.feature_df['forward_clearance'] = self.df[forward_sensors].mean(axis=1)
            
            # Left clearance
            left_sensors = [f'TRACK_EDGE_{i}' for i in [0, 1, 2] if f'TRACK_EDGE_{i}' in self.df.columns]
            if left_sensors:
                self.feature_df['left_clearance'] = self.df[left_sensors].mean(axis=1)
            
            # Right clearance
            right_sensors = [f'TRACK_EDGE_{i}' for i in [16, 17, 18] if f'TRACK_EDGE_{i}' in self.df.columns]
            if right_sensors:
                self.feature_df['right_clearance'] = self.df[right_sensors].mean(axis=1)
        
        # Behavioral consistency features (using rolling windows)
        window_size = 20
        self.feature_df['speed_stability'] = self.df['SPEED'].rolling(window_size).std().fillna(0)
        self.feature_df['steering_smoothness'] = self.df['STEERING'].rolling(window_size).std().fillna(0)
        self.feature_df['control_consistency'] = (
            self.df['ACCELERATION'].rolling(window_size).std().fillna(0) + 
            self.df['BRAKE'].rolling(window_size).std().fillna(0)
        ) / 2
        
        # Risk assessment features
        if 'min_distance' in self.feature_df.columns:
            self.feature_df['track_risk'] = abs(self.df['TRACK_POSITION']) * (1 / (self.feature_df['min_distance'] + 0.1))
        self.feature_df['cornering_risk'] = abs(self.df['STEERING']) * self.df['SPEED'] * abs(self.df['TRACK_POSITION'])
        
        print(f"Feature engineering completed. Added {len(self.feature_df.columns) - len(self.df.columns)} new features.")
        return self.feature_df
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("TORCS DATA ANALYSIS REPORT")
        print("="*60)
        
        # Dataset overview
        overview = self.generate_dataset_overview()
        
        # Track distribution
        print(f"\nTRACK TYPE DISTRIBUTION:")
        track_dist = self.df['track_type'].value_counts()
        for track_type, count in track_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {track_type:12s}: {count:6,} samples ({percentage:5.1f}%)")
        
        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        perf_metrics = self.calculate_performance_metrics()
        for track_type in perf_metrics.index:
            metrics = perf_metrics.loc[track_type]
            print(f"  {track_type:12s}: Speed={metrics['avg_speed']:5.1f}m/s, "
                  f"Steering={metrics['steering_consistency']:.3f}, "
                  f"Off-track={metrics['off_track_percentage']:4.1f}%")
        
        # Feature correlations
        print(f"\nKEY CORRELATIONS WITH SPEED:")
        _, speed_corr = self.perform_correlation_analysis()
        for feature, correlation in speed_corr.head(6).items():
            if feature != 'SPEED':
                print(f"  {feature:20s}: {correlation:6.4f}")
        
        # Data quality summary
        missing_count = self.df.isnull().sum().sum()
        off_track_count = (abs(self.df['TRACK_POSITION']) > 1).sum()
        
        print(f"\nDATA QUALITY:")
        print(f"  Missing values: {missing_count:,} ({missing_count/self.df.size*100:.3f}%)")
        print(f"  Off-track samples: {off_track_count:,} ({off_track_count/len(self.df)*100:.2f}%)")
        print(f"  Data completeness: {((self.df.size - missing_count)/self.df.size)*100:.2f}%")
        
        print("\n" + "="*60)


def main():
    """Main function for TORCS data analysis."""
    
    # Configuration
    DATA_PATH = "."  # Adjust to your data directory
    
    print("TORCS Autonomous Racing Data Analysis")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = TORCSDataAnalyzer(DATA_PATH)
        
        # Load datasets
        file_info = analyzer.load_all_datasets()
        
        # Data quality check
        analyzer.data_quality_check()
        
        # Clean and validate
        analyzer.clean_and_validate_data()
        
        # Perform analysis
        analyzer.perform_correlation_analysis()
        analyzer.analyze_racing_lines()
        
        # Create visualizations
        analyzer.create_comprehensive_visualizations(save_plots=True)
        
        # Normalize data
        analyzer.normalize_features()
        
        # Engineer features
        analyzer.engineer_features()
        
        # Generate final report
        analyzer.generate_report()
        
        # Save processed data
        print("\nSaving processed data...")
        analyzer.df.to_csv('torcs_cleaned_data.csv', index=False)
        if analyzer.normalized_df is not None:
            analyzer.normalized_df.to_csv('torcs_normalized_data.csv', index=False)
        if analyzer.feature_df is not None:
            analyzer.feature_df.to_csv('torcs_engineered_features.csv', index=False)
        
        print("\nAnalysis completed successfully!")
        print("Generated files:")
        print("  - torcs_comprehensive_analysis.png")
        print("  - torcs_cleaned_data.csv")
        print("  - torcs_normalized_data.csv")
        print("  - torcs_engineered_features.csv")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()