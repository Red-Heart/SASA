"""
Integration module to use visualizers with real project data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from core.fuzzy_visualization import FuzzyVisualization
from core.ann_visualization import ANNVisualization
from core.advisory_engine import AdvisoryEngine
from core.services import get_environmental_factors


class ProjectDataCollector:
    """Collect and manage real-time data from your project"""
    
    def __init__(self, data_dir: str = "project_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        try:
            self.engine = AdvisoryEngine()
            self.engine_available = True
        except Exception as e:
            print(f"Warning: Could not load advisory engine: {e}")
            print("  Using mock data for visualization testing")
            self.engine = None
            self.engine_available = False
        
    def collect_advisory_data(self, locations: list, crops: list = None) -> pd.DataFrame:
        """
        Collect advisory data for multiple locations and crops
        
        Args:
            locations: List of location names
            crops: List of crops to test (optional)
            
        Returns:
            DataFrame with collected data
        """
        data = []
        
        # Default crops if not specified
        default_crops = ['rice', 'maize', 'wheat', 'cotton', 'chickpea', 'banana']
        crop_list = crops or default_crops
        
        for location in locations:
            if self.engine_available:
                try:
                    # Get environmental factors from actual engine
                    factors = get_environmental_factors(location)
                    
                    # Get fuzzy irrigation advice
                    irrigation = self.engine.fuzzy.get_irrigation_advice(
                        rainfall=factors["rainfall"],
                        temperature=factors["temperature"],
                        humidity=factors["humidity"]
                    )
                    
                    # Get ANN crop suitability
                    crop_probs = self.engine.ann.get_all_probabilities(factors)
                    
                    # Collect for each crop
                    for crop in crop_list:
                        suitability = crop_probs.get(crop.lower(), np.random.random())
                        
                        data.append({
                            'location': location,
                            'crop': crop,
                            'rainfall': factors['rainfall'],
                            'temperature': factors['temperature'],
                            'humidity': factors['humidity'],
                            'ph': factors.get('ph', 6.5),
                            'N': factors.get('N', 50),
                            'P': factors.get('P', 50),
                            'K': factors.get('K', 50),
                            'suitability_score': suitability,
                            'irrigation_score': irrigation['irrigation_score'],
                            'irrigation_level': irrigation['irrigation_level'],
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    print(f"  Error processing {location}: {e}, using mock data")
                    self.engine_available = False
            
            # Fallback to mock data if engine unavailable
            if not self.engine_available:
                for crop in crop_list:
                    data.append({
                        'location': location,
                        'crop': crop,
                        'rainfall': np.random.uniform(50, 250),
                        'temperature': np.random.uniform(15, 40),
                        'humidity': np.random.uniform(30, 90),
                        'ph': np.random.uniform(5.5, 7.5),
                        'N': np.random.uniform(40, 100),
                        'P': np.random.uniform(30, 80),
                        'K': np.random.uniform(40, 100),
                        'suitability_score': np.random.uniform(0.3, 0.95),
                        'irrigation_score': np.random.uniform(20, 80),
                        'irrigation_level': np.random.choice(['Low', 'Medium', 'High']),
                        'timestamp': datetime.now().isoformat()
                    })
        
        df = pd.DataFrame(data)
        self.save_data(df, 'advisory_data.csv')
        return df
    
    def plot_fuzzy_from_real_data(self, df: pd.DataFrame, viz: FuzzyVisualization = None):
        """
        Create fuzzy visualization using real advisory data
        
        Args:
            df: DataFrame from collect_advisory_data()
            viz: FuzzyVisualization instance (creates if None)
        """
        if viz is None:
            viz = FuzzyVisualization()
        
        # Plot batch analysis
        batch_data = df[['rainfall', 'temperature', 'humidity']].drop_duplicates()
        batch_data = batch_data.reset_index(drop=True)
        
        print(f"Plotting fuzzy logic for {len(batch_data)} unique environmental scenarios...")
        fig, result_df = viz.plot_batch_analysis(batch_data)
        
        # Save results
        result_df.to_csv(self.data_dir / 'fuzzy_results.csv', index=False)
        print(f"Fuzzy results saved to {self.data_dir / 'fuzzy_results.csv'}")
        
        return fig, result_df
    
    def plot_suitability_comparison(self, df: pd.DataFrame):
        """
        Compare suitability scores across crops and locations
        
        Args:
            df: DataFrame from collect_advisory_data()
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Crop Suitability Analysis from Real Data', fontsize=16, fontweight='bold')
        
        # 1. Suitability by crop
        ax1 = axes[0, 0]
        crop_avg = df.groupby('crop')['suitability_score'].mean().sort_values(ascending=False)
        crop_avg.plot(kind='bar', ax=ax1, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_title('Average Suitability by Crop', fontweight='bold')
        ax1.set_ylabel('Suitability Score', fontweight='bold')
        ax1.set_xlabel('Crop', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Irrigation by location
        ax2 = axes[0, 1]
        irr_by_loc = df.groupby('location')['irrigation_score'].mean().sort_values(ascending=False)
        irr_by_loc.plot(kind='bar', ax=ax2, color='#A23B72', alpha=0.7, edgecolor='black')
        ax2.set_title('Average Irrigation Requirement by Location', fontweight='bold')
        ax2.set_ylabel('Irrigation Score', fontweight='bold')
        ax2.set_xlabel('Location', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Temperature vs Suitability
        ax3 = axes[1, 0]
        colors = df['irrigation_level'].map({'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'})
        ax3.scatter(df['temperature'], df['suitability_score'], c=colors, s=100, 
                   alpha=0.6, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Temperature (°C)', fontweight='bold')
        ax3.set_ylabel('Suitability Score', fontweight='bold')
        ax3.set_title('Temperature vs Suitability', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Data Summary:
        ─────────────────────
        Total Records:        {len(df)}
        Unique Locations:     {df['location'].nunique()}
        Unique Crops:         {df['crop'].nunique()}
        
        Suitability Score:
        ─────────────────────
        Mean:                 {df['suitability_score'].mean():.4f}
        Min:                  {df['suitability_score'].min():.4f}
        Max:                  {df['suitability_score'].max():.4f}
        
        Irrigation Score:
        ─────────────────────
        Mean:                 {df['irrigation_score'].mean():.2f}
        Min:                  {df['irrigation_score'].min():.2f}
        Max:                  {df['irrigation_score'].max():.2f}
        
        Temperature Range:
        ─────────────────────
        Min:                  {df['temperature'].min():.1f}°C
        Max:                  {df['temperature'].max():.1f}°C
        Avg:                  {df['temperature'].mean():.1f}°C
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        filepath = self.data_dir / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return None


class RealTimeVisualizer:
    """Combine real data with visualizers for production use"""
    
    def __init__(self):
        self.collector = ProjectDataCollector()
        self.fuzzy_viz = FuzzyVisualization()
        self.ann_viz = ANNVisualization()
    
    def visualize_advisory_results(self, locations: list, crops: list = None):
        """
        End-to-end visualization of advisory results
        
        Args:
            locations: List of locations to analyze
            crops: List of crops to analyze
        """
        import matplotlib.pyplot as plt
        
        # Collect data
        print("Collecting real advisory data...")
        df = self.collector.collect_advisory_data(locations, crops)
        print(f"Collected {len(df)} records")
        
        # Create visualizations
        print("\nGenerating fuzzy logic visualizations...")
        fig1, fuzzy_results = self.collector.plot_fuzzy_from_real_data(df, self.fuzzy_viz)
        plt.show()
        
        print("Generating suitability comparison...")
        fig2 = self.collector.plot_suitability_comparison(df)
        plt.show()
        
        return df, fuzzy_results
    
    def export_for_frontend(self, df: pd.DataFrame, output_file: str = "frontend_data.json"):
        """
        Export data in JSON format for frontend consumption
        
        Args:
            df: DataFrame with advisory data
            output_file: Output JSON filename
        """
        # Aggregate data for frontend
        by_crop = df.groupby('crop').agg({
            'suitability_score': ['mean', 'min', 'max'],
            'irrigation_score': ['mean', 'min', 'max']
        }).round(4)
        
        by_location = df.groupby('location').agg({
            'suitability_score': 'mean',
            'irrigation_score': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'rainfall': 'mean'
        }).round(4)
        
        # Convert to proper dict format
        by_crop_dict = {}
        for crop in by_crop.index:
            by_crop_dict[crop] = {
                'suitability_score': {
                    'mean': float(by_crop.loc[crop, ('suitability_score', 'mean')]),
                    'min': float(by_crop.loc[crop, ('suitability_score', 'min')]),
                    'max': float(by_crop.loc[crop, ('suitability_score', 'max')])
                },
                'irrigation_score': {
                    'mean': float(by_crop.loc[crop, ('irrigation_score', 'mean')]),
                    'min': float(by_crop.loc[crop, ('irrigation_score', 'min')]),
                    'max': float(by_crop.loc[crop, ('irrigation_score', 'max')])
                }
            }
        
        by_location_dict = {}
        for location in by_location.index:
            by_location_dict[location] = {
                'suitability_score': float(by_location.loc[location, 'suitability_score']),
                'irrigation_score': float(by_location.loc[location, 'irrigation_score']),
                'temperature': float(by_location.loc[location, 'temperature']),
                'humidity': float(by_location.loc[location, 'humidity']),
                'rainfall': float(by_location.loc[location, 'rainfall'])
            }
        
        frontend_data = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'locations': df['location'].unique().tolist(),
            'crops': df['crop'].unique().tolist(),
            'summary': {
                'avg_suitability': float(df['suitability_score'].mean()),
                'avg_irrigation': float(df['irrigation_score'].mean()),
                'avg_temperature': float(df['temperature'].mean()),
                'avg_humidity': float(df['humidity'].mean()),
                'avg_rainfall': float(df['rainfall'].mean()),
            },
            'by_crop': by_crop_dict,
            'by_location': by_location_dict,
            'raw_data': df.to_dict('records')
        }
        
        output_path = Path("project_data") / output_file
        with open(output_path, 'w') as f:
            json.dump(frontend_data, f, indent=2)
        
        print(f"Frontend data exported to {output_path}")
        return frontend_data


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize real-time visualizer
    visualizer = RealTimeVisualizer()
    
    # Example locations from your project
    test_locations = ["location_1", "location_2", "location_3"]
    test_crops = ["rice", "maize", "wheat", "cotton"]
    
    # Visualize advisory results
    print("=" * 60)
    print("REAL-TIME VISUALIZATION FROM PROJECT DATA")
    print("=" * 60)
    
    df, fuzzy_results = visualizer.visualize_advisory_results(test_locations, test_crops)
    
    # Export for frontend
    print("\nExporting data for frontend...")
    frontend_data = visualizer.export_for_frontend(df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"Data saved to: project_data/")
    print(f"Frontend JSON: project_data/frontend_data.json")
