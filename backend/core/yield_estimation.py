CROP_YIELD_INDEX = {
    # Cereals & fibers
    "rice": 1.00,
    "maize": 1.10,
    "cotton": 0.60,
    "jute": 0.65,

    # Pulses
    "chickpea": 0.80,
    "kidneybeans": 0.85,
    "pigeonpeas": 0.78,
    "mothbeans": 0.70,
    "mungbean": 0.82,
    "blackgram": 0.80,
    "lentil": 0.75,

    # Fruits
    "banana": 1.20,
    "mango": 1.00,
    "grapes": 1.10,
    "apple": 0.95,
    "orange": 0.98,
    "papaya": 1.15,
    "pomegranate": 0.90,
    "coconut": 1.05,

    # Cucurbits
    "watermelon": 1.10,
    "muskmelon": 1.05,

    # Plantation
    "coffee": 0.70
}

IRRIGATION_FACTOR = {
    "Low": 0.85,
    "Medium": 1.00,
    "High": 1.10
}

def estimate_yield_potential(crop, suitability, irrigation_level):
    base = CROP_YIELD_INDEX.get(crop, 0.8)
    irrigation_factor = IRRIGATION_FACTOR[irrigation_level]

    yield_index = base * suitability * irrigation_factor

    return round(yield_index, 3)