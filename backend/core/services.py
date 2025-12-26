import random
from typing import Dict


def get_environmental_factors(location: str) -> Dict[str, float]:

    # Climate
    temperature = round(random.uniform(26, 38), 1)   # °C
    humidity = round(random.uniform(40, 85), 1)      # %
    rainfall = round(random.uniform(20, 150), 1)     # mm

    # Soil parameters
    N = round(random.uniform(40, 120), 1)
    P = round(random.uniform(20, 60), 1)
    K = round(random.uniform(20, 80), 1)
    ph = round(random.uniform(5.5, 7.5), 2)

    return {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "N": N,
        "P": P,
        "K": K,
        "ph": ph
    }