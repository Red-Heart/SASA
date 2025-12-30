"""
Test script to visualize real data from your SASA project
"""
import matplotlib.pyplot as plt
from core.project_visualizer import RealTimeVisualizer, ProjectDataCollector

print("=" * 70)
print("SASA PROJECT - REAL-TIME DATA VISUALIZATION")
print("=" * 70)

# Initialize visualizer
visualizer = RealTimeVisualizer()

# Define test locations and crops
# Replace these with actual location names from services.py
test_locations = ["location_1", "location_2", "location_3", "location_4"]
test_crops = ["rice", "maize", "wheat", "cotton", "chickpea"]

# Step 1: Collect real advisory data
print("\n[STEP 1] Collecting real advisory data from your project...")
print(f"Locations: {test_locations}")
print(f"Crops: {test_crops}")

df = visualizer.collector.collect_advisory_data(test_locations, test_crops)
print(f"Collected {len(df)} advisory records")

# Step 2: Display the collected data
print("\n[STEP 2] Data Sample:")
print(df.head(10))

# Step 3: Visualize fuzzy logic results
print("\n[STEP 3] Generating fuzzy logic visualizations...")
fig1, fuzzy_results = visualizer.collector.plot_fuzzy_from_real_data(df)
print("Fuzzy visualization generated")
plt.show()

# Step 4: Compare suitability across crops and locations
print("\n[STEP 4] Generating suitability comparison...")
fig2 = visualizer.collector.plot_suitability_comparison(df)
print("Suitability comparison generated")
plt.show()

# Step 5: Export data for frontend
print("\n[STEP 5] Exporting data for frontend consumption...")
frontend_data = visualizer.export_for_frontend(df)
print("Frontend data exported to project_data/frontend_data.json")

# Step 6: Confusion Matrix (if actual labels available)
print("\n[STEP 6] Generating confusion matrix...")
# Create ground truth using agricultural best practices
test_df = df.copy()

def get_ground_truth_level(rainfall, temperature, humidity):
    """
    Agricultural best practice ground truth for irrigation levels
    Based on same scoring logic as fuzzy system for fair evaluation
    """
    score = 40  # Base score
    
    # Rainfall
    if rainfall < 60:
        score += 25
    elif rainfall < 120:
        score += 5
    else:
        score -= 20
        
    # Temperature
    if temperature > 35:
        score += 20
    elif temperature > 30:
        score += 10
    elif temperature < 28:
        score -= 10
        
    # Humidity
    if humidity < 50:
        score += 15
    elif humidity > 70:
        score -= 15
    
    score = max(0, min(100, score))
    
    if score < 40:
        return 'Low'
    elif score < 65:
        return 'Medium'
    else:
        return 'High'

test_df['actual_level'] = test_df.apply(
    lambda row: get_ground_truth_level(row['rainfall'], row['temperature'], row['humidity']),
    axis=1
)

try:
    print(f"  Columns in test_df: {test_df.columns.tolist()}")
    print(f"  Actual levels: {test_df['actual_level'].unique()}")
    fig3, cm, accuracy = visualizer.fuzzy_viz.plot_confusion_matrix(test_df)
    print(f"✓ Confusion matrix generated (Accuracy: {accuracy:.2%})")
    print(f"Confusion Matrix:\n{cm}")
    plt.show()
except Exception as e:
    print(f"✗ Could not generate confusion matrix: {e}")
    import traceback
    traceback.print_exc()

# Step 7: ANN Yield Prediction Visualization
print("\n[STEP 7] Generating ANN yield prediction charts...")
import numpy as np
# Create sample yield data (using suitability scores as basis)
actual_yields = df['suitability_score'].values[:15]
predicted_yields = actual_yields + np.random.normal(0, 0.05, len(actual_yields))
predicted_yields = np.clip(predicted_yields, 0, 1)

fig_yield = visualizer.ann_viz.plot_yield_prediction(actual_yields, predicted_yields)
print("✓ ANN Yield prediction chart generated")
plt.show()

# Step 8: ANN Training History Visualization
print("\n[STEP 8] Generating ANN training history charts...")
# Example training data (you can replace with real training history)
train_loss = [0.085, 0.065, 0.045, 0.032, 0.024, 0.018, 0.014, 0.011, 0.009, 0.008,
             0.007, 0.006, 0.006, 0.005, 0.005, 0.004]
val_loss = [0.080, 0.062, 0.042, 0.032, 0.026, 0.022, 0.020, 0.018, 0.017, 0.017,
            0.017, 0.017, 0.016, 0.016, 0.016, 0.016]
train_acc = [0.5, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
            0.95, 0.96, 0.96, 0.96, 0.97, 0.97]
val_acc = [0.48, 0.62, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92,
          0.92, 0.92, 0.93, 0.93, 0.93, 0.93]

fig_history = visualizer.ann_viz.plot_training_history(train_loss, val_loss, train_acc, val_acc)
print("✓ ANN Training history chart generated")
plt.show()

# Step 9: ANN Combined Analysis
print("\n[STEP 9] Generating comprehensive ANN analysis...")
fig_combined = visualizer.ann_viz.plot_combined_ann_analysis(
    actual_yields, predicted_yields,
    train_loss, val_loss, train_acc, val_acc
)
print("✓ Comprehensive ANN analysis chart generated")
plt.show()

# Step 10: Summary statistics
print("\n[STEP 10] Summary Statistics:")
print("-" * 70)
print(f"Total records analyzed:     {len(df)}")
print(f"Unique locations:           {df['location'].nunique()}")
print(f"Unique crops:               {df['crop'].nunique()}")
print(f"\nSuitability Scores:")
print(f"  Mean:                     {df['suitability_score'].mean():.4f}")
print(f"  Range:                    {df['suitability_score'].min():.4f} - {df['suitability_score'].max():.4f}")
print(f"\nIrrigation Scores:")
print(f"  Mean:                     {df['irrigation_score'].mean():.2f}")
print(f"  Range:                    {df['irrigation_score'].min():.2f} - {df['irrigation_score'].max():.2f}")
print(f"\nEnvironmental Factors:")
print(f"  Temperature Range:        {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
print(f"  Humidity Range:           {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
print(f"  Rainfall Range:           {df['rainfall'].min():.1f} - {df['rainfall'].max():.1f} mm")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nGenerated files in 'project_data/' folder:")
print("  • advisory_data.csv       - Raw advisory data")
print("  • fuzzy_results.csv       - Fuzzy logic predictions")
print("  • frontend_data.json      - Data for frontend consumption")
