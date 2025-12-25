import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyIrrigationSystem:
    def __init__(self):
        # Inputs
        self.rainfall = ctrl.Antecedent(np.arange(0, 301, 1), 'rainfall')
        self.temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

        # Output
        self.irrigation = ctrl.Consequent(np.arange(0, 101, 1), 'irrigation')

        self._define_membership_functions()
        self._define_rules()

        # Control system (stateless)
        self.irrigation_ctrl = ctrl.ControlSystem(self.rules)

    def _define_membership_functions(self):
        self.rainfall['low'] = fuzz.trimf(self.rainfall.universe, [0, 50, 100])
        self.rainfall['medium'] = fuzz.trimf(self.rainfall.universe, [80, 150, 220])
        self.rainfall['high'] = fuzz.trimf(self.rainfall.universe, [200, 250, 300])

        self.temperature['low'] = fuzz.trimf(self.temperature.universe, [0, 10, 20])
        self.temperature['medium'] = fuzz.trimf(self.temperature.universe, [15, 25, 35])
        self.temperature['high'] = fuzz.trimf(self.temperature.universe, [30, 40, 50])

        self.humidity['low'] = fuzz.trimf(self.humidity.universe, [0, 30, 50])
        self.humidity['medium'] = fuzz.trimf(self.humidity.universe, [40, 60, 80])
        self.humidity['high'] = fuzz.trimf(self.humidity.universe, [70, 85, 100])

        self.irrigation['low'] = fuzz.trimf(self.irrigation.universe, [0, 20, 40])
        self.irrigation['medium'] = fuzz.trimf(self.irrigation.universe, [30, 50, 70])
        self.irrigation['high'] = fuzz.trimf(self.irrigation.universe, [60, 80, 100])

    def _define_rules(self):
        self.rules = [
            ctrl.Rule(self.rainfall['low'] & self.temperature['high'], self.irrigation['high']),
            ctrl.Rule(self.rainfall['low'] & self.humidity['low'], self.irrigation['high']),
            ctrl.Rule(self.rainfall['medium'] & self.temperature['medium'], self.irrigation['medium']),
            ctrl.Rule(self.rainfall['high'], self.irrigation['low']),
            ctrl.Rule(self.humidity['high'] & self.rainfall['medium'], self.irrigation['low']),
            ctrl.Rule(self.temperature['low'] & self.rainfall['medium'], self.irrigation['low']),
            ctrl.Rule(self.temperature['medium'] & self.humidity['medium'], self.irrigation['medium']),
        ]

    def get_irrigation_advice(self, rainfall, temperature, humidity):
        # Create a fresh simulator per call
        simulator = ctrl.ControlSystemSimulation(self.irrigation_ctrl)

        simulator.input['rainfall'] = rainfall
        simulator.input['temperature'] = temperature
        simulator.input['humidity'] = humidity

        simulator.compute()

        score = simulator.output['irrigation']

        if score < 40:
            level = "Low"
        elif score < 70:
            level = "Medium"
        else:
            level = "High"

        return {
            "inputs": {
                "rainfall": rainfall,
                "temperature": temperature,
                "humidity": humidity
            },
            "irrigation_score": round(float(score), 2),
            "irrigation_level": level
        }