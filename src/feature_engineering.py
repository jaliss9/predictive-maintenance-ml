"""
Feature Engineering Module
Création des features pour la prédiction RUL
"""

import pandas as pd
import numpy as np


# Capteurs utiles pour le feature engineering
USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

# Capteurs principaux pour rolling statistics
ROLLING_SENSORS = [
    'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9', 'sensor_11',
    'sensor_12', 'sensor_14', 'sensor_17', 'sensor_20', 'sensor_21'
]


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Ajoute les moyennes mobiles et écarts-types mobiles pour les capteurs clés.
    
    Args:
        df: DataFrame avec les données capteurs
        window: Taille de la fenêtre pour le rolling (défaut: 5 cycles)
    
    Returns:
        DataFrame avec les nouvelles features
    """
    df = df.copy()
    
    for sensor in ROLLING_SENSORS:
        if sensor not in df.columns:
            continue
            
        # Moyenne mobile par moteur
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Écart-type mobile par moteur
        df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Remplacer les NaN par 0 (premiers cycles de chaque moteur)
    df = df.fillna(0)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Retourne la liste des colonnes à utiliser comme features.
    Exclut les colonnes non-features (unit_number, time_in_cycles, RUL, etc.)
    """
    exclude_cols = ['unit_number', 'time_in_cycles', 'max_cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prépare X (features) et y (target) pour l'entraînement.
    
    Returns:
        Tuple (X, y)
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['RUL']
    return X, y


if __name__ == "__main__":
    # Test du module
    from data_processing import load_train_data, add_rul_column, remove_useless_sensors
    
    df = load_train_data()
    df = add_rul_column(df)
    df = remove_useless_sensors(df)
    df = add_rolling_features(df)
    
    X, y = prepare_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Colonnes: {list(X.columns)}")