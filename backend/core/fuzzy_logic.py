import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyIrrigationSystem:
    def __init__(self):
        """Initialize irrigation advisory system using Fuzzy Logic"""
        self._setup_fuzzy_system()
    
    def _setup_fuzzy_system(self):
        """Setup fuzzy control system with variables and rules"""
        # Define input/output ranges (universe of discourse)
        self.rainfall_range = np.arange(0, 301, 1)
        self.temperature_range = np.arange(0, 51, 1)
        self.humidity_range = np.arange(0, 101, 1)
        self.irrigation_range = np.arange(0, 101, 1)
        
        # Define membership functions for rainfall (mm)
        self.rainfall_low = fuzz.trimf(self.rainfall_range, [0, 30, 60])
        self.rainfall_medium = fuzz.trimf(self.rainfall_range, [40, 90, 140])
        self.rainfall_high = fuzz.trimf(self.rainfall_range, [100, 200, 300])
        
        # Define membership functions for temperature (°C)
        self.temperature_low = fuzz.trimf(self.temperature_range, [0, 20, 28])
        self.temperature_medium = fuzz.trimf(self.temperature_range, [26, 32, 38])
        self.temperature_high = fuzz.trimf(self.temperature_range, [35, 42, 50])
        
        # Define membership functions for humidity (%)
        self.humidity_low = fuzz.trimf(self.humidity_range, [0, 25, 50])
        self.humidity_medium = fuzz.trimf(self.humidity_range, [40, 60, 80])
        self.humidity_high = fuzz.trimf(self.humidity_range, [70, 85, 100])
        
        # Define membership functions for irrigation output
        self.irrigation_low = fuzz.trimf(self.irrigation_range, [0, 20, 40])
        self.irrigation_medium = fuzz.trimf(self.irrigation_range, [30, 50, 70])
        self.irrigation_high = fuzz.trimf(self.irrigation_range, [60, 80, 100])

    def get_irrigation_advice(self, rainfall, temperature, humidity):
        """
        Calculate irrigation advice using fuzzy logic with direct computation
        
        Args:
            rainfall: Amount of rainfall in mm (0-300)
            temperature: Temperature in Celsius (0-50)
            humidity: Humidity percentage (0-100)
            
        Returns:
            Dict with inputs, irrigation_score, and irrigation_level
        """
        try:
            # Ensure inputs are within valid ranges
            rainfall = max(0, min(300, float(rainfall)))
            temperature = max(0, min(50, float(temperature)))
            humidity = max(0, min(100, float(humidity)))
            
            # Compute fuzzy membership values for inputs
            rainfall_low_val = fuzz.interp_membership(self.rainfall_range, self.rainfall_low, rainfall)
            rainfall_medium_val = fuzz.interp_membership(self.rainfall_range, self.rainfall_medium, rainfall)
            rainfall_high_val = fuzz.interp_membership(self.rainfall_range, self.rainfall_high, rainfall)
            
            temperature_low_val = fuzz.interp_membership(self.temperature_range, self.temperature_low, temperature)
            temperature_medium_val = fuzz.interp_membership(self.temperature_range, self.temperature_medium, temperature)
            temperature_high_val = fuzz.interp_membership(self.temperature_range, self.temperature_high, temperature)
            
            humidity_low_val = fuzz.interp_membership(self.humidity_range, self.humidity_low, humidity)
            humidity_medium_val = fuzz.interp_membership(self.humidity_range, self.humidity_medium, humidity)
            humidity_high_val = fuzz.interp_membership(self.humidity_range, self.humidity_high, humidity)
            
            # Apply fuzzy rules and collect output contributions
            irrigation_activation = np.zeros_like(self.irrigation_range, dtype=float)
            
            # HIGH IRRIGATION NEEDED rules
            activation = np.fmin(rainfall_low_val, humidity_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_high))
            
            activation = np.fmin(rainfall_low_val, temperature_high_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_high))
            
            activation = np.fmin(temperature_high_val, humidity_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_high))
            
            activation = np.fmin(np.fmin(rainfall_low_val, temperature_high_val), humidity_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_high))
            
            # LOW IRRIGATION NEEDED rules
            activation = np.fmin(rainfall_high_val, humidity_high_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_low))
            
            activation = np.fmin(rainfall_high_val, temperature_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_low))
            
            activation = np.fmin(humidity_high_val, temperature_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_low))
            
            # MEDIUM IRRIGATION NEEDED rules
            activation = np.fmin(rainfall_medium_val, temperature_medium_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_medium))
            
            activation = np.fmin(rainfall_medium_val, humidity_medium_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_medium))
            
            activation = np.fmin(temperature_medium_val, humidity_medium_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_medium))
            
            activation = np.fmin(rainfall_low_val, humidity_medium_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_medium))
            
            activation = np.fmin(rainfall_medium_val, temperature_low_val)
            irrigation_activation = np.fmax(irrigation_activation, 
                                            np.fmin(activation, self.irrigation_medium))
            
            # Defuzzify to get crisp output
            score = fuzz.defuzz(self.irrigation_range, irrigation_activation, 'centroid')
            
            # Handle case where defuzzification returns nan
            if np.isnan(score):
                score = self._fallback_score(rainfall, temperature, humidity)
            
        except Exception as e:
            print(f"Warning: Fuzzy logic computation failed ({e}), using fallback")
            score = self._fallback_score(rainfall, temperature, humidity)
        
        # Convert score to level
        if score < 40:
            level = "Low"
        elif score < 65:
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
    
    def _fallback_score(self, rainfall, temperature, humidity):
        """Fallback heuristic scoring when fuzzy logic fails"""
        score = 40
        
        if rainfall < 60:
            score += 25
        elif rainfall < 120:
            score += 5
        else:
            score -= 20
        
        if temperature > 35:
            score += 20
        elif temperature > 30:
            score += 10
        elif temperature < 28:
            score -= 10
        
        if humidity < 50:
            score += 15
        elif humidity > 70:
            score -= 15
        
        return max(0, min(100, score))