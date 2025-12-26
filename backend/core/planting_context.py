from datetime import datetime

# Malaysia-specific usual planting months (heuristic, not predictive)
USUAL_PLANTING_MONTHS = {
    # Cereals / Fibres
    "rice":        [5, 6, 7, 8],
    "maize":       [3, 4, 9, 10],
    "cotton":      [4, 5, 6],
    "jute":        [3, 4, 5],

    # Pulses
    "chickpea":    [10, 11],
    "kidneybeans": [3, 4],
    "pigeonpeas":  [5, 6],
    "mothbeans":   [6, 7],
    "mungbean":    [3, 4, 9],
    "blackgram":   [6, 7, 8],
    "lentil":      [10, 11],

    # Fruits
    "banana":      [1, 2, 3, 9, 10],
    "mango":       [4, 5, 6],
    "grapes":      [1, 2],
    "apple":       [12, 1],
    "orange":      [3, 4],
    "papaya":      [1, 2, 3],
    "pomegranate": [6, 7],
    "coconut":     [5, 6, 7],

    # Cucurbits
    "watermelon":  [2, 3, 4],
    "muskmelon":   [2, 3],

    # Plantation
    "coffee":      [5, 6],
}

SUITABILITY_THRESHOLD = 0.5


def evaluate_planting_context(
    crop: str,
    suitability_score: float,
    current_month: int | None = None
):
    """
    Classifies advisory into TT / TF / FT / FF categories.
    """

    crop = crop.lower()

    if current_month is None:
        current_month = datetime.now().month

    favourable = suitability_score >= SUITABILITY_THRESHOLD
    usual_months = USUAL_PLANTING_MONTHS.get(crop, [])
    usual_time = current_month in usual_months

    if favourable and usual_time:
        label = "Favourable and usually planted now"
        code = "TT"
    elif favourable and not usual_time:
        label = "Favourable but not usually planted now"
        code = "TF"
    elif not favourable and usual_time:
        label = "Not favourable but usually planted now"
        code = "FT"
    else:
        label = "Not favourable and not usually planted now"
        code = "FF"

    return {
        "context_code": code,
        "context_label": label,
        "current_month": current_month,
        "usual_planting_months": usual_months,
    }
