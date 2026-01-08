"""
Model Module
Entraînement et évaluation du modèle de prédiction RUL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Entraîne un modèle Random Forest pour prédire le RUL.
    
    Args:
        X: Features
        y: Target (RUL)
        test_size: Proportion du test set
        random_state: Seed pour reproductibilité
    
    Returns:
        Dict contenant le modèle, les métriques et les données de test
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Entraînement
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    return {
        'model': model,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': list(X.columns)
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Retourne l'importance des features triée par ordre décroissant."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return importance


def save_model(model, filepath: str = 'models/rf_model.joblib'):
    """Sauvegarde le modèle sur disque."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé: {filepath}")


def load_model(filepath: str = 'models/rf_model.joblib'):
    """Charge un modèle depuis le disque."""
    return joblib.load(filepath)


if __name__ == "__main__":
    # Test du module
    from data_processing import load_train_data, add_rul_column, remove_useless_sensors
    from feature_engineering import add_rolling_features, prepare_features
    
    # Pipeline complet
    df = load_train_data()
    df = add_rul_column(df)
    df = remove_useless_sensors(df)
    df = add_rolling_features(df)
    X, y = prepare_features(df)
    
    # Entraînement
    results = train_model(X, y)
    
    print("=== MÉTRIQUES ===")
    print(f"MAE: {results['metrics']['mae']:.2f} cycles")
    print(f"RMSE: {results['metrics']['rmse']:.2f} cycles")
    print(f"R²: {results['metrics']['r2']:.4f}")
    
    # Feature importance
    importance = get_feature_importance(results['model'], results['feature_names'])
    print("\n=== TOP 10 FEATURES ===")
    print(importance.head(10))
    
    # Sauvegarder
    save_model(results['model'])