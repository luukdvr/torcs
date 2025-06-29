import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TORCSStatisticalAnalyzer:
    """Statistical analysis class for TORCS data."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.track_edge_columns = [f'TRACK_EDGE_{i}' for i in range(19)]
        self.control_columns = ['ACCELERATION', 'BRAKE', 'STEERING']
        self.state_columns = ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS']
        
    def descriptive_statistics(self):
        """Comprehensive descriptive statistics."""
        print("=== DESCRIPTIVE STATISTICS ===\n")
        
        # General statistics
        stats_df = self.df.describe()
        print("Basic statistics for all numeric variables:")
        print(stats_df.round(3))
        
        # Distribution per track type
        print("\n=== STATISTICS PER TRACK TYPE ===")
        
        key_metrics = ['SPEED', 'ACCELERATION', 'BRAKE', 'STEERING', 'TRACK_POSITION']
        
        for metric in key_metrics:
            if metric in self.df.columns:
                print(f"\n{metric}:")
                track_stats = self.df.groupby('track_type')[metric].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(3)
                print(track_stats)
        
        return stats_df
    
    def normality_tests(self):
        """Test normality of data distributions."""
        print("\n=== NORMALITY TESTS ===\n")
        
        normality_results = {}
        key_features = self.control_columns + self.state_columns
        
        # Add center sensor if available
        if 'TRACK_EDGE_9' in self.df.columns:
            key_features.append('TRACK_EDGE_9')
        
        for feature in key_features:
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                # Shapiro-Wilk test (sample for large datasets)
                if len(data) > 5000:
                    sample_data = data.sample(5000, random_state=42)
                else:
                    sample_data = data
                
                shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                
                # Anderson-Darling test
                anderson_stat, critical_values, significance_levels = stats.anderson(data, dist='norm')
                
                # Skewness and Kurtosis
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                normality_results[feature] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'anderson_stat': anderson_stat,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'is_normal': shapiro_p > 0.05
                }
                
                print(f"{feature:20s}: Shapiro p={shapiro_p:.6f}, "
                      f"Skew={skewness:6.3f}, Kurt={kurtosis:6.3f}, "
                      f"Normal={'Yes' if shapiro_p > 0.05 else 'No'}")
        
        return normality_results
    
    def correlation_analysis_advanced(self):
        """Comprehensive correlation analysis with different methods."""
        print("\n=== ADVANCED CORRELATION ANALYSIS ===\n")
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Pearson correlation (linear relationships)
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation (monotonic relationships)
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Focus on SPEED as target variable
        speed_pearson = pearson_corr['SPEED'].abs().sort_values(ascending=False)
        speed_spearman = spearman_corr['SPEED'].abs().sort_values(ascending=False)
        
        print("TOP 10 PEARSON CORRELATIONS WITH SPEED:")
        for i, (feature, corr) in enumerate(speed_pearson.head(11).items()):
            if feature != 'SPEED':
                print(f"{i+1:2d}. {feature:25s}: {corr:7.4f}")
        
        print("\nTOP 10 SPEARMAN CORRELATIONS WITH SPEED:")
        for i, (feature, corr) in enumerate(speed_spearman.head(11).items()):
            if feature != 'SPEED':
                print(f"{i+1:2d}. {feature:25s}: {corr:7.4f}")
        
        # Mutual Information (non-linear relationships)
        print("\nMUTUAL INFORMATION ANALYSIS:")
        
        # Prepare data (remove NaN values)
        clean_df = numeric_df.dropna()
        if len(clean_df) > 10000:  # Sample for performance
            clean_df = clean_df.sample(10000, random_state=42)
        
        X = clean_df.drop(columns=['SPEED'])
        y = clean_df['SPEED']
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        print("TOP 10 MUTUAL INFORMATION SCORES:")
        for i, (feature, score) in enumerate(mi_results.head(10).items()):
            print(f"{i+1:2d}. {feature:25s}: {score:7.4f}")
        
        return pearson_corr, spearman_corr, mi_results
    
    def feature_importance_analysis(self):
        """Feature importance analysis with different methods."""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===\n")
        
        # Prepare clean dataset
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        if len(numeric_df) > 20000:  # Sample for performance
            numeric_df = numeric_df.sample(20000, random_state=42)
        
        X = numeric_df.drop(columns=['SPEED'])
        y = numeric_df['SPEED']
        
        importance_results = {}
        
        # 1. Univariate Statistical Tests (F-score)
        print("1. UNIVARIATE F-SCORE ANALYSIS:")
        f_selector = SelectKBest(score_func=f_regression, k='all')
        f_selector.fit(X, y)
        f_scores = pd.Series(f_selector.scores_, index=X.columns).sort_values(ascending=False)
        
        importance_results['f_scores'] = f_scores
        
        for i, (feature, score) in enumerate(f_scores.head(10).items()):
            print(f"{i+1:2d}. {feature:25s}: {score:10.2f}")
        
        # 2. Random Forest Feature Importance
        print("\n2. RANDOM FOREST FEATURE IMPORTANCE:")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        importance_results['random_forest'] = rf_importance
        
        for i, (feature, importance) in enumerate(rf_importance.head(10).items()):
            print(f"{i+1:2d}. {feature:25s}: {importance:7.4f}")
        
        # 3. Linear Regression Coefficients
        print("\n3. LINEAR REGRESSION COEFFICIENTS:")
        
        # Standardize features for fair comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        lr_importance = pd.Series(np.abs(lr.coef_), index=X.columns).sort_values(ascending=False)
        
        importance_results['linear_regression'] = lr_importance
        
        r2 = r2_score(y, lr.predict(X_scaled))
        print(f"Linear Regression R²: {r2:.4f}")
        
        for i, (feature, coef) in enumerate(lr_importance.head(10).items()):
            print(f"{i+1:2d}. {feature:25s}: {coef:7.4f}")
        
        return importance_results
    
    def racing_specific_analysis(self):
        """Racing-specific statistical analyses."""
        print("\n=== RACING-SPECIFIC ANALYSES ===\n")
        
        # 1. Cornering Performance Analysis
        print("1. CORNER PERFORMANCE ANALYSIS:")
        
        # Define cornering as high steering input
        cornering_threshold = 0.3
        cornering_data = self.df[abs(self.df['STEERING']) > cornering_threshold]
        straight_data = self.df[abs(self.df['STEERING']) <= cornering_threshold]
        
        print(f"Corner samples: {len(cornering_data):,} ({len(cornering_data)/len(self.df)*100:.1f}%)")
        print(f"Straight samples: {len(straight_data):,} ({len(straight_data)/len(self.df)*100:.1f}%)")
        
        # Statistical comparison
        corner_speed = cornering_data['SPEED'].mean()
        straight_speed = straight_data['SPEED'].mean()
        
        # T-test for speed differences
        t_stat, t_p = stats.ttest_ind(cornering_data['SPEED'], straight_data['SPEED'])
        
        print(f"Average corner speed: {corner_speed:.2f} m/s")
        print(f"Average straight speed: {straight_speed:.2f} m/s")
        print(f"Speed difference significant: {'Yes' if t_p < 0.05 else 'No'} (p={t_p:.6f})")
        
        # 2. Track Position Optimization Analysis
        print("\n2. TRACK POSITION OPTIMIZATION:")
        
        # Bin speeds and analyze optimal track positions
        speed_bins = pd.cut(self.df['SPEED'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        position_analysis = self.df.groupby(speed_bins, observed=True).agg({
            'TRACK_POSITION': ['mean', 'std', 'count'],
            'STEERING': lambda x: abs(x).mean()
        }).round(3)
        
        print("Optimal track position per speed range:")
        print(position_analysis)
        
        # 3. Sensor Utilization Analysis
        print("\n3. SENSOR UTILIZATION ANALYSIS:")
        
        # Analyze which sensors are most informative
        available_sensors = [col for col in self.track_edge_columns if col in self.df.columns]
        sensor_info = {}
        
        for col in available_sensors:
            i = int(col.split('_')[-1])
            angle = (i - 9) * 10  # Convert to degrees (-90 to +90)
            sensor_data = self.df[col].dropna()
            
            if len(sensor_data) > 0:
                sensor_info[f'Sensor_{angle:+03d}°'] = {
                    'mean_distance': sensor_data.mean(),
                    'std_distance': sensor_data.std(),
                    'min_distance': sensor_data.min(),
                    'usage_frequency': (sensor_data < 50).mean()  # Close obstacle frequency
                }
        
        if sensor_info:
            sensor_df = pd.DataFrame(sensor_info).T
            print("Sensor characteristics (distance in meters):")
            print(sensor_df.round(3))
        
        # 4. Control Input Efficiency Analysis
        print("\n4. CONTROL INPUT EFFICIENCY:")
        
        # Simultaneous brake/throttle (inefficient)
        simultaneous = ((self.df['ACCELERATION'] > 0.1) & (self.df['BRAKE'] > 0.1))
        efficiency_rate = (1 - simultaneous.mean()) * 100
        
        # Smooth control changes
        accel_smoothness = self.df['ACCELERATION'].diff().abs().mean()
        brake_smoothness = self.df['BRAKE'].diff().abs().mean()
        steering_smoothness = self.df['STEERING'].diff().abs().mean()
        
        print(f"Control efficiency rate: {efficiency_rate:.2f}%")
        print(f"Acceleration smoothness: {accel_smoothness:.4f}")
        print(f"Brake smoothness: {brake_smoothness:.4f}")
        print(f"Steering smoothness: {steering_smoothness:.4f}")
        
        return {
            'cornering_analysis': {
                'corner_speed': corner_speed,
                'straight_speed': straight_speed,
                't_statistic': t_stat,
                'p_value': t_p
            },
            'sensor_analysis': sensor_df if 'sensor_df' in locals() else None,
            'efficiency_metrics': {
                'efficiency_rate': efficiency_rate,
                'accel_smoothness': accel_smoothness,
                'brake_smoothness': brake_smoothness,
                'steering_smoothness': steering_smoothness
            }
        }
    
    def outlier_analysis(self):
        """Comprehensive outlier detection and analysis."""
        print("\n=== OUTLIER ANALYSIS ===\n")
        
        outlier_results = {}
        key_features = self.control_columns + self.state_columns
        
        for feature in key_features:
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_outliers = (z_scores > 3).sum()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                
                # Modified Z-score (more robust)
                median = data.median()
                mad = np.median(np.abs(data - median))
                if mad != 0:
                    modified_z_scores = 0.6745 * (data - median) / mad
                    modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                else:
                    modified_z_outliers = 0
                
                outlier_results[feature] = {
                    'total_samples': len(data),
                    'z_score_outliers': z_outliers,
                    'iqr_outliers': iqr_outliers,
                    'modified_z_outliers': modified_z_outliers,
                    'z_score_pct': (z_outliers / len(data)) * 100,
                    'iqr_pct': (iqr_outliers / len(data)) * 100,
                    'modified_z_pct': (modified_z_outliers / len(data)) * 100
                }
                
                print(f"{feature:20s}: Z-score={z_outliers:4d} ({z_outliers/len(data)*100:4.1f}%), "
                      f"IQR={iqr_outliers:4d} ({iqr_outliers/len(data)*100:4.1f}%), "
                      f"Mod-Z={modified_z_outliers:4d} ({modified_z_outliers/len(data)*100:4.1f}%)")
        
        return outlier_results
    
    def create_statistical_visualizations(self):
        """Create statistical visualizations."""
        print("\nCreating statistical visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('TORCS Statistical Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Distribution plots for key features
        key_features = ['SPEED', 'ACCELERATION', 'BRAKE', 'STEERING']
        for i, feature in enumerate(key_features):
            if i < 4 and feature in self.df.columns:
                row, col = i // 2, i % 2
                axes[0, col].hist(self.df[feature], bins=50, alpha=0.7, density=True)
                axes[0, col].set_title(f'{feature} Distribution')
                axes[0, col].set_ylabel('Density')
                axes[0, col].grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality
        if len(axes[0]) > 2:
            stats.probplot(self.df['SPEED'], dist="norm", plot=axes[0, 2])
            axes[0, 2].set_title('Speed Q-Q Plot (Normality Check)')
        
        # 3. Correlation heatmap (top features)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if 'SPEED' in numeric_df.columns:
            speed_corr = numeric_df.corr()['SPEED'].abs().sort_values(ascending=False)
            top_features = speed_corr.head(10).index
            
            corr_subset = numeric_df[top_features].corr()
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 0], square=True, cbar_kws={'label': 'Correlation'})
            axes[1, 0].set_title('Top Features Correlation Matrix')
        
        # 4. Boxplot per track type
        if 'track_type' in self.df.columns:
            track_types = self.df['track_type'].unique()[:4]  # Limit to 4 for readability
            track_speeds = [self.df[self.df['track_type'] == tt]['SPEED'] for tt in track_types]
            axes[1, 1].boxplot(track_speeds, labels=track_types)
            axes[1, 1].set_title('Speed Distribution by Track Type')
            axes[1, 1].set_ylabel('Speed (m/s)')
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Scatter plot with controls
        sample_df = self.df[['SPEED', 'ACCELERATION', 'BRAKE', 'STEERING']].sample(n=min(1000, len(self.df)))
        axes[1, 2].scatter(sample_df['ACCELERATION'], sample_df['SPEED'], alpha=0.6, s=10, label='Accel')
        axes[1, 2].set_xlabel('ACCELERATION')
        axes[1, 2].set_ylabel('SPEED')
        axes[1, 2].set_title('Speed vs Control Inputs')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 6-8. Additional plots
        for i in range(3):
            row, col = 2, i
            if col == 0:
                # Feature importance placeholder
                axes[row, col].text(0.5, 0.5, 'Feature Importance\n(Run feature_importance_analysis first)', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title('Feature Importance')
            elif col == 1:
                # Outlier visualization
                outlier_data = [self.df['SPEED'], self.df['ACCELERATION'], self.df['BRAKE'], abs(self.df['STEERING'])]
                axes[row, col].boxplot(outlier_data, labels=['Speed', 'Accel', 'Brake', 'Abs Steer'])
                axes[row, col].set_title('Outlier Detection (Boxplots)')
                axes[row, col].set_ylabel('Values')
                axes[row, col].grid(True, alpha=0.3)
            else:
                # Time series analysis (sample)
                sample_length = min(1000, len(self.df))
                time_axis = np.arange(sample_length) / 50
                axes[row, col].plot(time_axis, self.df['SPEED'].iloc[:sample_length], linewidth=1, label='Speed')
                axes[row, col].set_xlabel('Time (seconds)')
                axes[row, col].set_ylabel('Speed (m/s)')
                axes[row, col].set_title('Speed Time Series (Sample)')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('torcs_statistical_analysis.png', dpi=300, bbox_inches='tight')
        print("Statistical visualizations saved as 'torcs_statistical_analysis.png'")
        plt.show()
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report."""
        print("\n" + "="*70)
        print("TORCS STATISTICAL ANALYSIS REPORT")
        print("="*70)
        
        # Run all analyses
        desc_stats = self.descriptive_statistics()
        normality = self.normality_tests()
        correlations = self.correlation_analysis_advanced()
        importance = self.feature_importance_analysis()
        racing_analysis = self.racing_specific_analysis()
        outliers = self.outlier_analysis()
        
        # Summary conclusions
        print("\n=== KEY FINDINGS ===")
        print("1. DATA QUALITY:")
        print(f"   - Dataset contains {len(self.df):,} samples across {self.df['track_type'].nunique()} track types")
        print(f"   - Missing data: {self.df.isnull().sum().sum():,} values ({self.df.isnull().sum().sum()/self.df.size*100:.3f}%)")
        if 'TRACK_POSITION' in self.df.columns:
            print(f"   - Off-track samples: {(abs(self.df['TRACK_POSITION']) > 1).sum():,} ({(abs(self.df['TRACK_POSITION']) > 1).mean()*100:.2f}%)")
        
        print("\n2. PERFORMANCE INDICATORS:")
        speed_corr = correlations[0]['SPEED'].abs().sort_values(ascending=False)
        top_predictor = speed_corr.index[1]  # Exclude SPEED itself
        print(f"   - Primary speed predictor: {top_predictor} (r={speed_corr[top_predictor]:.4f})")
        print(f"   - Average speed: {self.df['SPEED'].mean():.2f} ± {self.df['SPEED'].std():.2f} m/s")
        print(f"   - Speed range: {self.df['SPEED'].min():.1f} - {self.df['SPEED'].max():.1f} m/s")
        
        print("\n3. RACING INSIGHTS:")
        corner_stats = racing_analysis['cornering_analysis']
        print(f"   - Corner vs straight speed difference: {corner_stats['straight_speed'] - corner_stats['corner_speed']:.2f} m/s")
        print(f"   - Control efficiency: {racing_analysis['efficiency_metrics']['efficiency_rate']:.1f}%")
        print(f"   - Steering smoothness: {racing_analysis['efficiency_metrics']['steering_smoothness']:.4f}")
        
        print("\n4. FEATURE RECOMMENDATIONS:")
        rf_importance = importance['random_forest']
        print("   Top 5 features for ML model:")
        for i, (feature, imp) in enumerate(rf_importance.head(5).items()):
            print(f"   {i+1}. {feature} (importance: {imp:.4f})")
        
        print("\n" + "="*70)
        
        return {
            'descriptive_stats': desc_stats,
            'normality_tests': normality,
            'correlations': correlations,
            'feature_importance': importance,
            'racing_analysis': racing_analysis,
            'outlier_analysis': outliers
        }


def run_statistical_analysis(csv_file_or_dataframe):
    """Run complete statistical analysis."""
    
    if isinstance(csv_file_or_dataframe, str):
        # If it's a file path
        df = pd.read_csv(csv_file_or_dataframe)
        print(f"Loaded data from {csv_file_or_dataframe}")
    else:
        # If it's already a DataFrame
        df = csv_file_or_dataframe
        print("Using provided DataFrame")
    
    # Initialize analyzer
    analyzer = TORCSStatisticalAnalyzer(df)
    
    # Run complete analysis
    results = analyzer.generate_statistical_report()
    
    # Create visualizations
    analyzer.create_statistical_visualizations()
    
    return analyzer, results


# Example usage
if __name__ == "__main__":
    # If you run this script standalone
    print("TORCS Statistical Analysis Module")
    print("Import this module and use run_statistical_analysis(your_dataframe)")
    print("Example:")
    print("  from torcs_statistical_analysis import run_statistical_analysis")
    print("  analyzer, results = run_statistical_analysis('your_data.csv')")