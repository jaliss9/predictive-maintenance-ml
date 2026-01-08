"""
Data Processing Module
Chargement et préparation des données NASA C-MAPSS
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Noms des colonnes du dataset
COLUMN_NAMES = (
    ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Capteurs à variance nulle (inutiles)
USELESS_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                   'sensor_16', 'sensor_18', 'sensor_19']

# Plafond RUL
RUL_CAP = 125


def load_train_data(data_path: str = 'data/train_FD001.txt') -> pd.DataFrame:
    """Charge les données d'entraînement."""
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df


def load_test_data(data_path: str = 'data/test_FD001.txt') -> pd.DataFrame:
    """Charge les données de test."""
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df


def load_rul_data(data_path: str = 'data/RUL_FD001.txt') -> pd.Series:
    """Charge les valeurs RUL réelles pour le test set."""
    rul = pd.read_csv(data_path, header=None, names=['RUL'])
    return rul['RUL']


def add_rul_column(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Ajoute la colonne RUL (Remaining Useful Life) au dataframe.
    RUL = max_cycle_du_moteur - cycle_actuel, plafonné à `cap`.
    """
    df = df.copy()
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df['RUL'] = df['RUL'].clip(upper=cap)
    
    return df


def remove_useless_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les capteurs à variance nulle."""
    cols_to_drop = [col for col in USELESS_SENSORS if col in df.columns]
    return df.drop(columns=cols_to_drop)


if __name__ == "__main__":
    # Test du module
    df = load_train_data()
    print(f"Données chargées: {df.shape}")
    
    df = add_rul_column(df)
    print(f"RUL ajouté: {df['RUL'].describe()}")
    
    df = remove_useless_sensors(df)
    print(f"Après suppression capteurs inutiles: {df.shape}")