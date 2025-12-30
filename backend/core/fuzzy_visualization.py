import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .fuzzy_logic import FuzzyIrrigationSystem
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


class FuzzyVisualization:
    """Visualize fuzzy logic outputs using pandas and matplotlib"""
    
    def __init__(self):
        self.fuzzy_system = FuzzyIrrigationSystem()
    
    def plot_single_output(self, rainfall: float, temperature: float, humidity: float):
        """
        Plot a single fuzzy logic output with input values
        
        Args:
            rainfall: Rainfall amount (0-300)
            temperature: Temperature (0-50)
            humidity: Humidity percentage (0-100)
        """
        try:
            result = self.fuzzy_system.get_irrigation_advice(rainfall, temperature, humidity)
        except Exception as e:
            print(f"Warning: Could not compute fuzzy logic: {e}")
            # Use fallback values
            result = {
                "inputs": {"rainfall": rainfall, "temperature": temperature, "humidity": humidity},
                "irrigation_score": 50.0,
                "irrigation_level": "Medium"
            }
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fuzzy Logic Irrigation System Output', fontsize=16, fontweight='bold')
        
        # 1. Input values bar chart
        inputs = result['inputs']
        ax1 = axes[0, 0]
        input_names = list(inputs.keys())
        input_values = list(inputs.values())
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        ax1.bar(input_names, input_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Value', fontweight='bold')
        ax1.set_title('Input Parameters', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(input_values):
            ax1.text(i, v + 5, f'{v}', ha='center', fontweight='bold')
        
        # 2. Irrigation score gauge
        ax2 = axes[0, 1]
        score = result['irrigation_score']
        level = result['irrigation_level']
        color_map = {'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'}
        gauge_color = color_map.get(level, '#FFD93D')
        ax2.barh(['Irrigation'], [score], color=gauge_color, alpha=0.7, edgecolor='black', height=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Score', fontweight='bold')
        ax2.set_title(f'Irrigation Level: {level}', fontweight='bold')
        ax2.text(score/2, 0, f'{score}', ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        
        # 3. Membership function visualizations
        ax3 = axes[1, 0]
        self._plot_membership_functions(ax3, rainfall, temperature, humidity)
        ax3.set_title('Input Membership Values', fontweight='bold')
        
        # 4. Data summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Rainfall', f'{rainfall} mm'],
            ['Temperature', f'{temperature}°C'],
            ['Humidity', f'{humidity}%'],
            ['Irrigation Score', f'{score}'],
            ['Irrigation Level', level]
        ]
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def plot_batch_analysis(self, dataframe: pd.DataFrame):
        """
        Plot fuzzy logic results for a batch of inputs
        
        Args:
            dataframe: DataFrame with columns ['rainfall', 'temperature', 'humidity']
        """
        # Ensure required columns exist
        required_cols = ['rainfall', 'temperature', 'humidity']
        if not all(col in dataframe.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Compute irrigation advice for each row
        results = []
        for idx, row in dataframe.iterrows():
            try:
                result = self.fuzzy_system.get_irrigation_advice(
                    row['rainfall'], row['temperature'], row['humidity']
                )
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not compute fuzzy logic for row {idx}: {e}")
                # Use fallback values
                results.append({
                    "inputs": {
                        "rainfall": row['rainfall'],
                        "temperature": row['temperature'],
                        "humidity": row['humidity']
                    },
                    "irrigation_score": 50.0,  # Default middle value
                    "irrigation_level": "Medium"
                })
        
        # Convert to DataFrame
        result_df = pd.json_normalize(results)
        result_df['irrigation_level_numeric'] = result_df['irrigation_level'].map(
            {'Low': 1, 'Medium': 2, 'High': 3}
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fuzzy Logic Batch Analysis', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Rainfall vs Irrigation Score
        ax1 = axes[0, 0]
        colors = result_df['irrigation_level_numeric'].map({1: '#FF6B6B', 2: '#FFD93D', 3: '#6BCB77'})
        ax1.scatter(result_df['inputs.rainfall'], result_df['irrigation_score'], 
                   c=colors, s=100, alpha=0.6, edgecolors='black')
        ax1.set_xlabel('Rainfall (mm)', fontweight='bold')
        ax1.set_ylabel('Irrigation Score', fontweight='bold')
        ax1.set_title('Rainfall vs Irrigation Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot: Temperature vs Irrigation Score
        ax2 = axes[0, 1]
        ax2.scatter(result_df['inputs.temperature'], result_df['irrigation_score'],
                   c=colors, s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Temperature (°C)', fontweight='bold')
        ax2.set_ylabel('Irrigation Score', fontweight='bold')
        ax2.set_title('Temperature vs Irrigation Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution of irrigation levels
        ax3 = axes[1, 0]
        level_counts = result_df['irrigation_level'].value_counts()
        colors_map = {'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'}
        colors_pie = [colors_map.get(level, '#999999') for level in level_counts.index]
        explode = tuple([0.05] * len(level_counts))  # Dynamic explode based on actual categories
        ax3.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, explode=explode)
        ax3.set_title('Irrigation Level Distribution', fontweight='bold')
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        stats = [
            ['Statistic', 'Value'],
            ['Total Records', len(result_df)],
            ['Avg Irrigation Score', f'{result_df["irrigation_score"].mean():.2f}'],
            ['Min Irrigation Score', f'{result_df["irrigation_score"].min():.2f}'],
            ['Max Irrigation Score', f'{result_df["irrigation_score"].max():.2f}'],
            ['Most Common Level', result_df['irrigation_level'].mode().values[0]]
        ]
        table = ax4.table(cellText=stats, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig, result_df
    
    def plot_confusion_matrix(self, dataframe: pd.DataFrame, actual_col: str = 'actual_level'):
        """
        Plot confusion matrix comparing predicted vs actual irrigation levels
        
        Args:
            dataframe: DataFrame with columns ['rainfall', 'temperature', 'humidity', 'actual_level']
            actual_col: Column name containing actual irrigation levels ('Low', 'Medium', 'High')
        """
        if actual_col not in dataframe.columns:
            raise ValueError(f"DataFrame must contain '{actual_col}' column with actual irrigation levels")
        
        required_cols = ['rainfall', 'temperature', 'humidity']
        if not all(col in dataframe.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Get predictions
        predictions = []
        for idx, row in dataframe.iterrows():
            try:
                result = self.fuzzy_system.get_irrigation_advice(
                    row['rainfall'], row['temperature'], row['humidity']
                )
                predictions.append(result['irrigation_level'])
            except Exception as e:
                print(f"  Warning: Could not compute fuzzy logic for row {idx}: {e}")
                # Use fallback prediction based on heuristics
                temp = row['temperature']
                humidity = row['humidity']
                rainfall = row['rainfall']
                
                # Simple fallback logic
                if rainfall > 200:
                    predictions.append('Low')
                elif temp > 35:
                    predictions.append('High')
                elif humidity > 75:
                    predictions.append('Low')
                else:
                    predictions.append('Medium')
        
        actual = dataframe[actual_col].tolist()
        
        # Compute confusion matrix
        labels = ['Low', 'Medium', 'High']
        cm = confusion_matrix(actual, predictions, labels=labels)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Confusion Matrix Heatmap
        ax1 = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                   yticklabels=labels, ax=ax1, cbar_kws={'label': 'Count'},
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_xlabel('Predicted Level', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Actual Level', fontweight='bold', fontsize=11)
        ax1.set_title('Confusion Matrix', fontweight='bold')
        
        # Plot 2: Metrics Summary
        ax2 = axes[1]
        ax2.axis('tight')
        ax2.axis('off')
        
        # Calculate metrics
        accuracy = accuracy_score(actual, predictions)
        report = classification_report(actual, predictions, labels=labels, output_dict=True, zero_division=0)
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Overall Accuracy', f'{accuracy:.2%}'],
            ['', ''],
            ['Class', 'Precision | Recall | F1'],
            ['Low', f'{report["Low"]["precision"]:.2f} | {report["Low"]["recall"]:.2f} | {report["Low"]["f1-score"]:.2f}'],
            ['Medium', f'{report["Medium"]["precision"]:.2f} | {report["Medium"]["recall"]:.2f} | {report["Medium"]["f1-score"]:.2f}'],
            ['High', f'{report["High"]["precision"]:.2f} | {report["High"]["recall"]:.2f} | {report["High"]["f1-score"]:.2f}'],
            ['', ''],
            ['Weighted Avg', f'{report["weighted avg"]["precision"]:.2f} | {report["weighted avg"]["recall"]:.2f} | {report["weighted avg"]["f1-score"]:.2f}'],
        ]
        
        table = ax2.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header rows
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(2):
            table[(3, i)].set_facecolor('#E3F2FD')
            table[(3, i)].set_text_props(weight='bold')
        
        plt.tight_layout()
        
        # Print detailed report
        print("\n" + "="*60)
        print("CONFUSION MATRIX CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(actual, predictions, labels=labels, zero_division=0))
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("="*60 + "\n")
        
        return fig, cm, accuracy
    
    def _plot_membership_functions(self, ax, rainfall, temperature, humidity):
        """Helper to plot membership function values"""
        # Create a simple bar chart showing which membership functions are active
        membership_data = {
            'Rainfall Low': 1 if rainfall < 100 else 0,
            'Rainfall Medium': 1 if 80 < rainfall < 220 else 0,
            'Rainfall High': 1 if rainfall > 200 else 0,
            'Temp Low': 1 if temperature < 20 else 0,
            'Temp Medium': 1 if 15 < temperature < 35 else 0,
            'Temp High': 1 if temperature > 30 else 0,
            'Humidity Low': 1 if humidity < 50 else 0,
            'Humidity Medium': 1 if 40 < humidity < 80 else 0,
            'Humidity High': 1 if humidity > 70 else 0,
        }
        
        names = list(membership_data.keys())
        values = list(membership_data.values())
        colors = ['#66B2FF' if v == 1 else '#CCCCCC' for v in values]
        
        ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('Active', fontweight='bold')
        ax.set_xticks([0, 1])


# Example usage
if __name__ == "__main__":
    viz = FuzzyVisualization()
    
    # Single plot example
    fig = viz.plot_single_output(rainfall=50, temperature=30, humidity=45)
    plt.show()
    
    # Batch analysis example
    sample_data = pd.DataFrame({
        'rainfall': [25, 50, 100, 150, 200, 250, 100, 75],
        'temperature': [15, 20, 25, 30, 35, 40, 28, 22],
        'humidity': [30, 45, 60, 75, 85, 90, 55, 50]
    })
    
    fig, result_df = viz.plot_batch_analysis(sample_data)
    print("\nBatch Analysis Results:")
    print(result_df)
    plt.show()
    
    # Confusion matrix example
    test_data = pd.DataFrame({
        'rainfall': [25, 50, 100, 150, 200, 250, 100, 75, 30, 180, 120, 60],
        'temperature': [15, 20, 25, 30, 35, 40, 28, 22, 18, 38, 26, 24],
        'humidity': [30, 45, 60, 75, 85, 90, 55, 50, 35, 88, 65, 48],
        'actual_level': ['Low', 'Low', 'Medium', 'Medium', 'High', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium', 'Low']
    })
    
    fig, cm, accuracy = viz.plot_confusion_matrix(test_data)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2%}")
    plt.show()
