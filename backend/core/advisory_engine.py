from core.fuzzy_logic import FuzzyIrrigationSystem
from core.ann_inference import ANNSuitability
from core.yield_estimation import estimate_yield_potential
from core.planting_context import evaluate_planting_context

class AdvisoryEngine:
    def __init__(self):
        self.fuzzy = FuzzyIrrigationSystem()
        self.ann = ANNSuitability()

    def generate_advisory(self, features: dict, crop: str = None):
        # 1. Fuzzy irrigation
        irrigation = self.fuzzy.get_irrigation_advice(
            rainfall=features["rainfall"],
            temperature=features["temperature"],
            humidity=features["humidity"]
        )

        irrigation_level = irrigation["irrigation_level"]

        # 2. ANN probabilities for all crops
        crop_probs = self.ann.get_all_probabilities(features)

        # Crop specified
        if crop:
            crop = crop.lower()
            suitability = float(crop_probs.get(crop, 0.0))

            yield_potential = float(estimate_yield_potential(
                crop,
                suitability,
                irrigation_level
            ))
            planting_context = evaluate_planting_context(
                crop=crop,
                suitability_score=suitability
            )
            return {
                "mode": "crop_specific",
                "crop": crop,
                "suitability_score": round(suitability, 3),
                "irrigation": irrigation,
                "yield_potential": yield_potential,
                "planting_context": planting_context
            }

        # Crop not specified
        results = []

        for crop_name, prob in crop_probs.items():
            prob_f = float(prob)
            yield_potential = float(estimate_yield_potential(
                crop_name,
                prob_f,
                irrigation_level
            ))
            planting_context = evaluate_planting_context(
                crop=crop_name,
                suitability_score=prob_f
            )
            results.append({
                "crop": crop_name,
                "suitability_score": round(prob_f, 3),
                "yield_potential": yield_potential,
                "planting_context": planting_context
            })

        # Sort by yield potential
        top_5 = sorted(
            results,
            key=lambda x: x["yield_potential"],
            reverse=True
        )[:5]

        return {
            "mode": "crop_recommendation",
            "irrigation": irrigation,
            "top_5_crops": top_5
        }