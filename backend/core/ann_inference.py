import joblib
import pandas as pd
from pathlib import Path

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

class ANNSuitability:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        self.model = joblib.load(model_dir / "ann_model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.encoder = joblib.load(model_dir / "crop_encoder.pkl")

    def get_all_probabilities(self, features: dict):
        df = pd.DataFrame([[features[f] for f in FEATURE_ORDER]],
                        columns=FEATURE_ORDER)

        x_scaled = self.scaler.transform(df)
        probs = self.model.predict_proba(x_scaled)[0]

        return dict(zip(self.encoder.classes_, probs))