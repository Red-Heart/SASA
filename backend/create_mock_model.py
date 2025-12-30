"""
Generate a mock trained ANN model for testing visualization
This creates dummy models that mimic the structure of real trained models
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.neural_network import MLPClassifier

print("Creating mock trained models for SASA project...")

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# 1. Create a mock scaler
print("  Creating scaler...")
scaler = StandardScaler()
# Fit it with dummy data that matches your features
dummy_features = np.array([
    [50, 25, 6.5, 100, 50, 50],  # N, P, K, temperature, humidity, ph, rainfall
    [60, 30, 7.0, 30, 60, 80],
    [40, 20, 6.0, 35, 70, 120],
    [70, 40, 7.5, 20, 40, 50],
    [55, 35, 6.8, 28, 65, 100],
])
scaler.fit(dummy_features)
joblib.dump(scaler, models_dir / "scaler.pkl")
print("    ✓ Scaler saved")

# 2. Create a mock crop encoder
print("  Creating crop encoder...")
crops = ['rice', 'maize', 'wheat', 'cotton', 'chickpea', 'kidneybeans', 
         'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil',
         'banana', 'mango', 'grapes', 'apple', 'orange', 'papaya', 
         'pomegranate', 'coconut', 'watermelon', 'muskmelon', 'coffee', 'jute']

encoder = LabelBinarizer()
encoder.fit(crops)
joblib.dump(encoder, models_dir / "crop_encoder.pkl")
print(f"    ✓ Encoder saved ({len(crops)} crop classes)")

# 3. Create a mock ANN model
print("  Creating mock ANN model...")
# The model should predict probabilities for each crop
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
    early_stopping=False
)

# Train with dummy data
np.random.seed(42)
X_dummy = np.random.randn(100, 6)  # 6 features (N, P, K, temp, humidity, ph, rainfall)
y_dummy = np.random.choice(crops, 100)

model.fit(X_dummy, y_dummy)
joblib.dump(model, models_dir / "ann_model.pkl")
print("    ✓ ANN Model saved")

print("\n" + "="*60)
print("✓ Mock trained models created successfully!")
print("="*60)
print("\nFiles created:")
print(f"  • {models_dir / 'ann_model.pkl'}")
print(f"  • {models_dir / 'scaler.pkl'}")
print(f"  • {models_dir / 'crop_encoder.pkl'}")
print("\nNow you can run: python test_real_data_viz.py")
