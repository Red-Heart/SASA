"""Quick test script to display fuzzy visualization outputs"""
import pandas as pd
import matplotlib.pyplot as plt
from core.fuzzy_visualization import FuzzyVisualization

# Initialize visualizer
viz = FuzzyVisualization()

# Test 1: Single prediction
print("Generating single prediction plot...")
fig1 = viz.plot_single_output(rainfall=50, temperature=30, humidity=45)
plt.show()

# Test 2: Batch analysis with multiple scenarios
print("Generating batch analysis plot...")
sample_data = pd.DataFrame({
    'rainfall': [25, 50, 100, 150, 200, 250, 100, 75],
    'temperature': [15, 20, 25, 30, 35, 40, 28, 22],
    'humidity': [30, 45, 60, 75, 85, 90, 55, 50]
})

fig2, result_df = viz.plot_batch_analysis(sample_data)
print("\nBatch Analysis Results:")
print(result_df)
plt.show()

# Test 3: Confusion Matrix
print("Generating confusion matrix plot...")
test_data = pd.DataFrame({
    'rainfall': [25, 50, 100, 150, 200, 250, 100, 75, 30, 180, 120, 60],
    'temperature': [15, 20, 25, 30, 35, 40, 28, 22, 18, 38, 26, 24],
    'humidity': [30, 45, 60, 75, 85, 90, 55, 50, 35, 88, 65, 48],
    'actual_level': ['Low', 'Low', 'Medium', 'Medium', 'High', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium', 'Low']
})

fig3, cm, accuracy = viz.plot_confusion_matrix(test_data)
print(f"\nConfusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2%}")
plt.show()
