import joblib
import pandas as pd
from pathlib import Path
import warnings

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

class ANNSuitability:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Models are in backend/models/, parent of parent directory
            model_dir = Path(__file__).parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        try:
            # Try with unsafe=True to handle numpy version mismatch
            self.model = joblib.load(model_dir / "ann_model.pkl", _load_numpy_version='1.26')
        except:
            try:
                # Fallback: try without version specification
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = joblib.load(model_dir / "ann_model.pkl")
            except Exception as e:
                raise Exception(f"Could not load model: {e}")
        
        try:
            self.scaler = joblib.load(model_dir / "scaler.pkl")
        except:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scaler = joblib.load(model_dir / "scaler.pkl")
        
        try:
            self.encoder = joblib.load(model_dir / "crop_encoder.pkl")
        except:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.encoder = joblib.load(model_dir / "crop_encoder.pkl")

    def get_all_probabilities(self, features: dict):
        df = pd.DataFrame([[features[f] for f in FEATURE_ORDER]],
                        columns=FEATURE_ORDER)

        x_scaled = self.scaler.transform(df)
        probs = self.model.predict_proba(x_scaled)[0]

        return dict(zip(self.encoder.classes_, probs))