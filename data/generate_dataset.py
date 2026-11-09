"""
Dataset Generation Script for Fertilizer Recommendation System
Generates a realistic synthetic dataset adhering to agronomic guidelines
for soil nutrients (N, P, K), environmental factors (Temp, Humidity, Moisture),
soil types, crop types, and target fertilizer classes.
"""

import os
import random
import numpy as np
import pandas as pd

# Set random seeds for exact reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']

CROP_TYPES = [
    'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 
    'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'
]

FERTILIZERS = [
    'Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26'
]

# Agronomic logic profiles for target fertilizer classes:
# Each fertilizer has typical soil nutrient deficiency profiles and environmental conditions
# Urea: Needs high Nitrogen supplementation when N is low
# DAP: High Phosphorus requirement when P is low and N is moderate
# 14-35-14: High P requirement with balanced N & K
# 28-28: High N and P requirement with low/adequate K
# 17-17-17: Balanced N-P-K depletion
# 20-20: Moderate N and P depletion
# 10-26-26: High P and high K demand (potassium-rich)
FERTILIZER_PROFILES = {
    'Urea': {
        'n_range': (10, 28),     # Soil is deficient in Nitrogen
        'p_range': (20, 50),     # Adequate/moderate Phosphorus
        'k_range': (15, 45),     # Adequate Potassium
        'temp_range': (24, 38),
        'humidity_range': (50, 80),
        'moisture_range': (30, 65),
        'preferred_soils': ['Loamy', 'Black', 'Clayey', 'Sandy', 'Red'],
        'preferred_crops': ['Paddy', 'Sugarcane', 'Wheat', 'Maize', 'Cotton']
    },
    'DAP': {
        'n_range': (20, 45),     # Moderate N
        'p_range': (8, 22),      # Severely deficient in Phosphorus
        'k_range': (15, 40),     # Adequate Potassium
        'temp_range': (20, 34),
        'humidity_range': (45, 75),
        'moisture_range': (25, 60),
        'preferred_soils': ['Black', 'Loamy', 'Clayey', 'Red'],
        'preferred_crops': ['Wheat', 'Paddy', 'Cotton', 'Pulses', 'Ground Nuts']
    },
    '14-35-14': {
        'n_range': (12, 28),
        'p_range': (10, 25),     # Low P
        'k_range': (10, 25),     # Low K
        'temp_range': (22, 35),
        'humidity_range': (52, 78),
        'moisture_range': (30, 55),
        'preferred_soils': ['Sandy', 'Red', 'Loamy'],
        'preferred_crops': ['Oil seeds', 'Ground Nuts', 'Cotton', 'Millets']
    },
    '28-28': {
        'n_range': (12, 26),     # Deficient in N
        'p_range': (12, 26),     # Deficient in P
        'k_range': (25, 55),     # Adequate K
        'temp_range': (23, 36),
        'humidity_range': (55, 82),
        'moisture_range': (35, 65),
        'preferred_soils': ['Black', 'Loamy', 'Clayey'],
        'preferred_crops': ['Sugarcane', 'Maize', 'Tobacco', 'Paddy']
    },
    '17-17-17': {
        'n_range': (15, 30),     # Balanced deficiency
        'p_range': (15, 30),
        'k_range': (15, 30),
        'temp_range': (21, 33),
        'humidity_range': (50, 75),
        'moisture_range': (30, 60),
        'preferred_soils': ['Loamy', 'Red', 'Sandy', 'Clayey'],
        'preferred_crops': ['Barley', 'Wheat', 'Maize', 'Millets', 'Pulses']
    },
    '20-20': {
        'n_range': (18, 35),
        'p_range': (18, 35),
        'k_range': (28, 55),
        'temp_range': (22, 36),
        'humidity_range': (48, 72),
        'moisture_range': (28, 58),
        'preferred_soils': ['Black', 'Loamy', 'Red'],
        'preferred_crops': ['Cotton', 'Sugarcane', 'Millets', 'Wheat']
    },
    '10-26-26': {
        'n_range': (25, 50),     # Adequate N
        'p_range': (10, 22),     # Low P
        'k_range': (8, 20),      # Low K (needs heavy Potash)
        'temp_range': (24, 37),
        'humidity_range': (55, 85),
        'moisture_range': (32, 65),
        'preferred_soils': ['Sandy', 'Loamy', 'Clayey', 'Red'],
        'preferred_crops': ['Ground Nuts', 'Oil seeds', 'Pulses', 'Tobacco']
    }
}


def generate_dataset(num_samples_per_class=95, output_path="data/fertilizer_dataset.csv"):
    """
    Generates balanced synthetic fertilizer dataset (~665 samples) with realistic
    agronomic properties, missing values, and slight label noise.
    """
    records = []
    
    for fert, prof in FERTILIZER_PROFILES.items():
        for _ in range(num_samples_per_class):
            temp = round(np.random.uniform(*prof['temp_range']), 1)
            humidity = round(np.random.uniform(*prof['humidity_range']), 1)
            moisture = round(np.random.uniform(*prof['moisture_range']), 1)
            
            # 80% preference for typical soil/crop, 20% random for realistic variance
            if np.random.rand() < 0.8:
                soil = np.random.choice(prof['preferred_soils'])
            else:
                soil = np.random.choice(SOIL_TYPES)
                
            if np.random.rand() < 0.8:
                crop = np.random.choice(prof['preferred_crops'])
            else:
                crop = np.random.choice(CROP_TYPES)
                
            nitrogen = int(np.random.uniform(*prof['n_range']))
            phosphorous = int(np.random.uniform(*prof['p_range']))
            potassium = int(np.random.uniform(*prof['k_range']))
            
            # Introduce minor class noise (approx 2% chance of alternate label)
            label = fert
            if np.random.rand() < 0.02:
                label = np.random.choice([f for f in FERTILIZERS if f != fert])
                
            records.append({
                'Temperature': temp,
                'Humidity': humidity,
                'Moisture': moisture,
                'Soil_Type': soil,
                'Crop_Type': crop,
                'Nitrogen': nitrogen,
                'Potassium': potassium,
                'Phosphorous': phosphorous,
                'Fertilizer_Name': label
            })
            
    df = pd.DataFrame(records)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Inject missing values (approx 2.5% missing in numeric features)
    numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in numeric_cols:
        mask = np.random.rand(len(df)) < 0.025
        df.loc[mask, col] = np.nan
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print("Class distribution:")
    print(df['Fertilizer_Name'].value_counts())
    print("\nMissing values injected:")
    print(df.isna().sum())
    return df


if __name__ == "__main__":
    generate_dataset()
