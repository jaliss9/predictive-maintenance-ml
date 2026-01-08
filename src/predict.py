"""
Predict Module
Fonctions de prédiction pour l'application Streamlit
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data_processing import COLUMN_NAMES, USELESS_SENSORS, RUL_CAP
from feature_engineering import add_rolling_features, ROLLING_SENSORS


def prepare_engine_data(df: pd.DataFrame, engine_id: int) -> pd.DataFrame:
    """
    Prépare les données d'un moteur spécifique pour la prédiction.
    
    Args:
        df: DataFrame complet
        engine_id: Numéro du moteur
    
    Returns:
        DataFrame avec les features pour ce moteur
    """
    engine_data = df[df['unit_number'] == engine_id].copy()
    return engine_data


def predict_rul(model, engine_data: pd.DataFrame, feature_names: list) -> np.ndarray:
    """
    Prédit le RUL pour chaque cycle d'un moteur.
    
    Args:
        model: Modèle entraîné
        engine_data: Données du moteur
        feature_names: Liste des features utilisées par le modèle
    
    Returns:
        Array des prédictions RUL
    """
    X = engine_data[feature_names]
    predictions = model.predict(X)
    return predictions


def get_current_rul(model, engine_data: pd.DataFrame, feature_names: list) -> dict:
    """
    Retourne la prédiction RUL pour le dernier cycle connu d'un moteur.
    
    Returns:
        Dict avec RUL prédit, cycle actuel, et statut d'alerte
    """
    predictions = predict_rul(model, engine_data, feature_names)
    current_rul = predictions[-1]
    current_cycle = engine_data['time_in_cycles'].iloc[-1]
    
    # Niveaux d'alerte
    if current_rul <= 15:
        alert_level = 'CRITIQUE'
        alert_color = 'red'
    elif current_rul <= 30:
        alert_level = 'ATTENTION'
        alert_color = 'orange'
    elif current_rul <= 50:
        alert_level = 'SURVEILLANCE'
        alert_color = 'yellow'
    else:
        alert_level = 'NORMAL'
        alert_color = 'green'
    
    return {
        'predicted_rul': round(current_rul, 1),
        'current_cycle': int(current_cycle),
        'alert_level': alert_level,
        'alert_color': alert_color,
        'all_predictions': predictions
    }


def get_sensor_trends(engine_data: pd.DataFrame) -> dict:
    """
    Calcule les tendances des capteurs clés pour un moteur.
    
    Returns:
        Dict avec les données de tendance pour chaque capteur
    """
    key_sensors = ['sensor_4', 'sensor_9', 'sensor_14', 'sensor_3']
    trends = {}
    
    for sensor in key_sensors:
        if sensor in engine_data.columns:
            trends[sensor] = {
                'values': engine_data[sensor].values,
                'cycles': engine_data['time_in_cycles'].values,
                'current': engine_data[sensor].iloc[-1],
                'mean': engine_data[sensor].mean(),
                'trend': 'up' if engine_data[sensor].iloc[-1] > engine_data[sensor].iloc[0] else 'down'
            }
    
    return trends


def get_fleet_status(model, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Retourne le statut RUL de tous les moteurs de la flotte.
    
    Returns:
        DataFrame avec le RUL prédit pour chaque moteur
    """
    fleet_status = []
    
    for engine_id in df['unit_number'].unique():
        engine_data = prepare_engine_data(df, engine_id)
        status = get_current_rul(model, engine_data, feature_names)
        
        fleet_status.append({
            'engine_id': engine_id,
            'current_cycle': status['current_cycle'],
            'predicted_rul': status['predicted_rul'],
            'alert_level': status['alert_level']
        })
    
    return pd.DataFrame(fleet_status).sort_values('predicted_rul')


if __name__ == "__main__":
    # Test du module
    from data_processing import load_train_data, add_rul_column, remove_useless_sensors
    from feature_engineering import add_rolling_features, get_feature_columns
    from model import load_model
    
    # Charger les données
    df = load_train_data()
    df = add_rul_column(df)
    df = remove_useless_sensors(df)
    df = add_rolling_features(df)
    
    feature_names = get_feature_columns(df)
    
    # Charger le modèle (doit être entraîné d'abord)
    try:
        model = load_model()
        
        # Test sur moteur 1
        engine_data = prepare_engine_data(df, engine_id=1)
        status = get_current_rul(model, engine_data, feature_names)
        
        print(f"=== MOTEUR 1 ===")
        print(f"Cycle actuel: {status['current_cycle']}")
        print(f"RUL prédit: {status['predicted_rul']} cycles")
        print(f"Alerte: {status['alert_level']}")
        
    except FileNotFoundError:
        print("Modèle non trouvé. Entraînez d'abord avec: python src/model.py")