"""Test script for ANN visualization outputs"""
import matplotlib.pyplot as plt
from core.ann_visualization import ANNVisualization

# Initialize visualizer
viz = ANNVisualization()

# Example data
actual_yields = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
predicted_yields = [0.12, 0.18, 0.32, 0.38, 0.52, 0.62, 0.68, 0.82, 0.88, 0.98,
                   0.14, 0.24, 0.36, 0.42, 0.58, 0.64, 0.72, 0.86, 0.92]

train_loss = [0.085, 0.065, 0.045, 0.032, 0.024, 0.018, 0.014, 0.011, 0.009, 0.008,
             0.007, 0.006, 0.006, 0.005, 0.005, 0.004]
val_loss = [0.080, 0.062, 0.042, 0.032, 0.026, 0.022, 0.020, 0.018, 0.017, 0.017,
            0.017, 0.017, 0.016, 0.016, 0.016, 0.016]
train_acc = [0.5, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
            0.95, 0.96, 0.96, 0.96, 0.97, 0.97]
val_acc = [0.48, 0.62, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92,
          0.92, 0.92, 0.93, 0.93, 0.93, 0.93]

# Test 1: Yield Prediction Plot
print("1. Generating yield prediction plot...")
fig1 = viz.plot_yield_prediction(actual_yields, predicted_yields)
plt.show()

# Test 2: Training History Plot (Loss + Accuracy)
print("2. Generating training history plot...")
fig2 = viz.plot_training_history(train_loss, val_loss, train_acc, val_acc)
plt.show()

# Test 3: Combined Analysis
print("3. Generating combined ANN analysis plot...")
fig3 = viz.plot_combined_ann_analysis(actual_yields, predicted_yields,
                                     train_loss, val_loss, train_acc, val_acc)
plt.show()

print("\nAll visualizations completed!")
