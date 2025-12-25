from fuzzy_logic import FuzzyIrrigationSystem

fuzzy_system = FuzzyIrrigationSystem()
result = fuzzy_system.get_irrigation_advice(
    rainfall=60,
    temperature=40,
    humidity=30
)

print(result)