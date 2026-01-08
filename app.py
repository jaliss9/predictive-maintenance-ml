"""
Predictive Maintenance - Railway Traction Motors
Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing import load_train_data, add_rul_column, remove_useless_sensors
from feature_engineering import add_rolling_features, get_feature_columns
from model import load_model, get_feature_importance
from predict import prepare_engine_data, get_current_rul, get_sensor_trends


st.set_page_config(
    page_title="Railway Predictive Maintenance",
    layout="wide"
)

# ----------------------------
# CONFIG GRAPHIQUES STATIQUES
# ----------------------------
STATIC_CONFIG = {
    'staticPlot': True,
    'displayModeBar': False
}

# ----------------------------
# TRADUCTIONS
# ----------------------------
TRANSLATIONS = {
    'fr': {
        'title': "Maintenance Prédictive : Moteurs de Traction Ferroviaire",
        'problem_text': """
        **Le problème** : Dans le transport ferroviaire, une panne de moteur de traction 
        entraîne des retards, des coûts de réparation élevés et des risques pour la disponibilité de la flotte.
        
        **La solution** : Prédire la **durée de vie restante** (RUL - Remaining Useful Life) 
        à partir des données capteurs embarqués (température, vibrations, pression, courant). 
        Cela permet de planifier la maintenance en atelier avant la défaillance en ligne.
        
        **Ce projet** : Un modèle Machine Learning entraîné sur des données de dégradation 
        de moteurs (dataset NASA C-MAPSS, représentatif des patterns de dégradation industriels). 
        Le modèle prédit combien de cycles d'opération restent avant la maintenance requise.
        
        *Application : flottes TGV, TER, locomotives, métros.*
        """,
        'live_prediction': "Prédiction en direct",
        'select_train': "Sélectionner une rame",
        'select_cycle': "Sélectionner le cycle d'analyse",
        'train_unit': "Rame",
        'current_cycle': "Cycle analysé",
        'predicted_rul': "RUL prédit",
        'cycles': "cycles",
        'status': "Statut",
        'alert_labels': {
            'CRITIQUE': 'Maintenance immédiate requise',
            'ATTENTION': 'Planifier maintenance sous 2 semaines',
            'SURVEILLANCE': 'Surveillance renforcée',
            'NORMAL': 'Opérationnel'
        },
        'alert_colors': {
            'CRITIQUE': '#e74c3c',
            'ATTENTION': '#f39c12',
            'SURVEILLANCE': '#f1c40f',
            'NORMAL': '#27ae60'
        },
        'chart_rul_predicted': "RUL prédit par le modèle",
        'chart_rul_actual': "RUL réel (données historiques)",
        'chart_threshold': "Seuil de maintenance préventive",
        'chart_current_cycle': "Cycle analysé",
        'xaxis_km': "Kilométrage (x1000 km)",
        'yaxis_cycles_remaining': "Cycles restants avant révision",
        'understand_chart': "Comprendre ce graphique",
        'chart_explanation_rul': """
        - **Ligne bleue** : Prédiction du modèle à chaque relevé kilométrique. Estimation de la durée de vie restante du moteur de traction.
        - **Ligne verte pointillée** : Valeur réelle issue des données historiques de maintenance.
        - **Ligne rouge horizontale** : Seuil de maintenance préventive. En dessous de 30 cycles, la rame doit être programmée en atelier.
        - **Ligne verticale violette** : Cycle actuellement sélectionné pour l'analyse.
        - **Axe X** : Kilométrage parcouru (en milliers de km).
        - **Axe Y** : Cycles restants avant révision majeure (0 = défaillance).
        
        Un bon modèle = la ligne bleue suit la ligne verte. Cela permet d'anticiper les entrées en atelier.
        """,
        'model_performance': "Performance du modèle",
        'metrics': "Métriques",
        'mae_caption': "Excellente précision : erreur moyenne de seulement 11 400 km sur des moteurs parcourant 200 000+ km.",
        'rmse_caption': "Erreurs importantes rares : le modèle reste fiable même sur les cas difficiles.",
        'r2_caption': "84% de précision : le modèle capture très bien les patterns de dégradation.",
        'predictions_vs_reality': "Prédictions vs Réalité (ensemble de la flotte)",
        'predictions': "Prédictions",
        'perfect_prediction': "Prédiction parfaite",
        'xaxis_rul_actual': "RUL réel (cycles)",
        'yaxis_rul_predicted': "RUL prédit (cycles)",
        'chart_explanation_scatter': """
        Chaque point représente une prédiction pour une rame à un instant donné.
        - **Axe X** : Durée de vie restante réelle (issue des données de maintenance)
        - **Axe Y** : Prédiction du modèle
        - **Ligne rouge** : Prédiction parfaite (prédit = réel)
        
        Plus les points sont concentrés autour de la ligne rouge, plus le modèle est fiable.
        """,
        'predictive_sensors': "Capteurs les plus prédictifs",
        'sensor_labels': {
            'sensor_4_rolling_mean': 'Température bobinage (moy.)',
            'sensor_9_rolling_mean': 'Vibrations palier (moy.)',
            'sensor_11_rolling_mean': 'Courant stator (moy.)',
            'sensor_14_rolling_mean': 'Pression huile (moy.)',
            'sensor_7_rolling_mean': 'Température huile (moy.)',
            'sensor_12_rolling_mean': 'Vitesse rotation (moy.)',
            'sensor_21_rolling_mean': 'Température ambiante (moy.)'
        },
        'importance_xaxis': "Importance dans la prédiction",
        'chart_explanation_importance': """
        Le modèle identifie automatiquement quels capteurs sont les plus prédictifs de la dégradation.
        
        - **Température bobinage** : Indicateur principal de l'usure du moteur de traction.
        - **Vibrations palier** : Signe d'usure mécanique des roulements.
        - **Courant stator** : Anomalies électriques détectables avant défaillance.
        
        Ces informations permettent de prioriser les capteurs à surveiller en exploitation.
        """,
        'pipeline': "Pipeline de traitement",
        'pipeline_steps': [
            ("1. Acquisition", ["Données capteurs embarqués", "21 mesures par cycle", "100 rames simulées"]),
            ("2. Nettoyage", ["Suppression capteurs défaillants", "Calcul du RUL historique", "Plafonnement à 125 cycles"]),
            ("3. Features", ["Moyennes mobiles (5 cycles)", "Écarts-types mobiles", "37 variables finales"]),
            ("4. Modèle ML", ["Random Forest", "100 arbres de décision", "Validation croisée"]),
            ("5. Déploiement", ["API de prédiction", "Dashboard temps réel", "Alertes automatiques"])
        ],
        'glossary_title': "Glossaire et contexte",
        'glossary': "Glossaire",
        'glossary_table': """
        | Terme | Définition |
        |-------|------------|
        | **RUL** | Remaining Useful Life. Durée de vie restante avant maintenance. |
        | **Cycle** | Unité d'opération (ici ~1000 km parcourus). |
        | **MAE** | Mean Absolute Error. Erreur moyenne de prédiction. |
        | **R²** | Coefficient de détermination (1 = parfait). |
        | **Rolling mean** | Moyenne glissante pour lisser le bruit des capteurs. |
        | **Maintenance prédictive** | Maintenance basée sur l'état réel de l'équipement, non sur un calendrier fixe. |
        """,
        'tech_stack': "Stack technique",
        'tech_stack_content': """
        - **Langage** : Python 3.11
        - **Traitement** : Pandas, NumPy
        - **Machine Learning** : Scikit-learn
        - **Visualisation** : Plotly
        - **Interface** : Streamlit
        - **Données** : NASA C-MAPSS (dégradation moteurs)
        
        *Ce projet démontre l'application des techniques de maintenance prédictive 
        utilisées dans l'industrie ferroviaire.*
        """,
        'error_msg': "Erreur",
        'error_info': "Lancez d'abord : python src/model.py"
    },
    'en': {
        'title': "Predictive Maintenance: Railway Traction Motors",
        'problem_text': """
        **The problem**: In rail transport, a traction motor failure 
        causes delays, high repair costs, and fleet availability risks.
        
        **The solution**: Predict the **Remaining Useful Life** (RUL) 
        using onboard sensor data (temperature, vibrations, pressure, current). 
        This enables scheduling maintenance before in-service failure.
        
        **This project**: A Machine Learning model trained on motor degradation data 
        (NASA C-MAPSS dataset, representative of industrial degradation patterns). 
        The model predicts how many operation cycles remain before required maintenance.
        
        *Application: high-speed trains, regional trains, locomotives, metros.*
        """,
        'live_prediction': "Live Prediction",
        'select_train': "Select a train unit",
        'select_cycle': "Select analysis cycle",
        'train_unit': "Unit",
        'current_cycle': "Analyzed cycle",
        'predicted_rul': "Predicted RUL",
        'cycles': "cycles",
        'status': "Status",
        'alert_labels': {
            'CRITIQUE': 'Immediate maintenance required',
            'ATTENTION': 'Schedule maintenance within 2 weeks',
            'SURVEILLANCE': 'Enhanced monitoring',
            'NORMAL': 'Operational'
        },
        'alert_colors': {
            'CRITIQUE': '#e74c3c',
            'ATTENTION': '#f39c12',
            'SURVEILLANCE': '#f1c40f',
            'NORMAL': '#27ae60'
        },
        'chart_rul_predicted': "Model predicted RUL",
        'chart_rul_actual': "Actual RUL (historical data)",
        'chart_threshold': "Preventive maintenance threshold",
        'chart_current_cycle': "Analyzed cycle",
        'xaxis_km': "Mileage (x1000 km)",
        'yaxis_cycles_remaining': "Cycles remaining before overhaul",
        'understand_chart': "Understanding this chart",
        'chart_explanation_rul': """
        - **Blue line**: Model prediction at each mileage reading. Estimated remaining life of the traction motor.
        - **Dashed green line**: Actual value from historical maintenance data.
        - **Red horizontal line**: Preventive maintenance threshold. Below 30 cycles, the unit should be scheduled for workshop.
        - **Purple vertical line**: Currently selected cycle for analysis.
        - **X-axis**: Mileage traveled (in thousands of km).
        - **Y-axis**: Cycles remaining before major overhaul (0 = failure).
        
        A good model = the blue line follows the green line. This enables anticipating workshop entries.
        """,
        'model_performance': "Model Performance",
        'metrics': "Metrics",
        'mae_caption': "Excellent accuracy: average error of only 11,400 km on motors running 200,000+ km.",
        'rmse_caption': "Large errors are rare: the model remains reliable even on difficult cases.",
        'r2_caption': "84% accuracy: the model captures degradation patterns very well.",
        'predictions_vs_reality': "Predictions vs Reality (entire fleet)",
        'predictions': "Predictions",
        'perfect_prediction': "Perfect prediction",
        'xaxis_rul_actual': "Actual RUL (cycles)",
        'yaxis_rul_predicted': "Predicted RUL (cycles)",
        'chart_explanation_scatter': """
        Each point represents a prediction for a train unit at a given time.
        - **X-axis**: Actual remaining life (from maintenance data)
        - **Y-axis**: Model prediction
        - **Red line**: Perfect prediction (predicted = actual)
        
        The closer points are to the red line, the more reliable the model.
        """,
        'predictive_sensors': "Most Predictive Sensors",
        'sensor_labels': {
            'sensor_4_rolling_mean': 'Winding temperature (avg.)',
            'sensor_9_rolling_mean': 'Bearing vibrations (avg.)',
            'sensor_11_rolling_mean': 'Stator current (avg.)',
            'sensor_14_rolling_mean': 'Oil pressure (avg.)',
            'sensor_7_rolling_mean': 'Oil temperature (avg.)',
            'sensor_12_rolling_mean': 'Rotation speed (avg.)',
            'sensor_21_rolling_mean': 'Ambient temperature (avg.)'
        },
        'importance_xaxis': "Importance in prediction",
        'chart_explanation_importance': """
        The model automatically identifies which sensors are most predictive of degradation.
        
        - **Winding temperature**: Main indicator of traction motor wear.
        - **Bearing vibrations**: Sign of mechanical bearing wear.
        - **Stator current**: Electrical anomalies detectable before failure.
        
        This information helps prioritize which sensors to monitor in operation.
        """,
        'pipeline': "Processing Pipeline",
        'pipeline_steps': [
            ("1. Acquisition", ["Onboard sensor data", "21 measurements per cycle", "100 simulated units"]),
            ("2. Cleaning", ["Remove faulty sensors", "Calculate historical RUL", "Cap at 125 cycles"]),
            ("3. Features", ["Rolling averages (5 cycles)", "Rolling std deviations", "37 final variables"]),
            ("4. ML Model", ["Random Forest", "100 decision trees", "Cross-validation"]),
            ("5. Deployment", ["Prediction API", "Real-time dashboard", "Automatic alerts"])
        ],
        'glossary_title': "Glossary and Context",
        'glossary': "Glossary",
        'glossary_table': """
        | Term | Definition |
        |------|------------|
        | **RUL** | Remaining Useful Life. Time remaining before maintenance. |
        | **Cycle** | Operation unit (here ~1000 km traveled). |
        | **MAE** | Mean Absolute Error. Average prediction error. |
        | **R²** | Coefficient of determination (1 = perfect). |
        | **Rolling mean** | Moving average to smooth sensor noise. |
        | **Predictive maintenance** | Maintenance based on actual equipment condition, not fixed schedule. |
        """,
        'tech_stack': "Tech Stack",
        'tech_stack_content': """
        - **Language**: Python 3.11
        - **Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-learn
        - **Visualization**: Plotly
        - **Interface**: Streamlit
        - **Data**: NASA C-MAPSS (motor degradation)
        
        *This project demonstrates predictive maintenance techniques 
        used in the railway industry.*
        """,
        'error_msg': "Error",
        'error_info': "First run: python src/model.py"
    }
}


@st.cache_data
def load_data():
    df = load_train_data()
    df = add_rul_column(df)
    df = remove_useless_sensors(df)
    df = add_rolling_features(df)
    return df


@st.cache_resource
def get_model_and_results():
    model = load_model()
    return model


@st.cache_data
def get_scatter_data(_model, df, feature_names):
    """Prépare les données du scatter plot avec cache."""
    X = df[feature_names]
    y_true = df['RUL'].values
    y_pred = _model.predict(X)
    
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(y_true), size=2000, replace=False)
    
    return y_true[sample_idx], y_pred[sample_idx]


def get_status_at_cycle(model, engine_data, feature_names, cycle_idx):
    """Retourne le statut à un cycle spécifique."""
    row = engine_data.iloc[[cycle_idx]]
    X = row[feature_names]
    predicted_rul = model.predict(X)[0]
    current_cycle = row['time_in_cycles'].values[0]
    
    if predicted_rul <= 15:
        alert_level = 'CRITIQUE'
    elif predicted_rul <= 30:
        alert_level = 'ATTENTION'
    elif predicted_rul <= 50:
        alert_level = 'SURVEILLANCE'
    else:
        alert_level = 'NORMAL'
    
    return {
        'predicted_rul': round(predicted_rul, 1),
        'current_cycle': int(current_cycle),
        'alert_level': alert_level
    }


def main():
    # Sélecteur de langue en haut à droite
    col_spacer, col_lang = st.columns([5, 1])
    
    with col_lang:
        lang = st.selectbox(
            "Language",
            ['en', 'fr'],
            format_func=lambda x: 'English' if x == 'en' else 'Français'
        )
    
    t = TRANSLATIONS[lang]
    
    # ----------------------------
    # SECTION 1 : LE PROBLÈME
    # ----------------------------
    st.title(t['title'])
    st.markdown(t['problem_text'])
    st.divider()
    
    # ----------------------------
    # SECTION 2 : PRÉDICTION EN DIRECT
    # ----------------------------
    st.header(t['live_prediction'])
    
    try:
        df = load_data()
        model = get_model_and_results()
        feature_names = get_feature_columns(df)
    except Exception as e:
        st.error(f"{t['error_msg']}: {e}")
        st.info(t['error_info'])
        return
    
    # Sélection rame et cycle
    col_select_train, col_select_cycle = st.columns([1, 2])
    
    with col_select_train:
        engine_ids = sorted(df['unit_number'].unique())
        selected_engine = st.selectbox(
            t['select_train'],
            engine_ids,
            format_func=lambda x: f"{t['train_unit']} {x:03d}"
        )
    
    engine_data = prepare_engine_data(df, selected_engine)
    max_cycles = len(engine_data)
    
    with col_select_cycle:
        selected_cycle_idx = st.slider(
            t['select_cycle'],
            min_value=0,
            max_value=max_cycles - 1,
            value=max_cycles // 2,
            format=f"Cycle %d"
        )
    
    # Statut au cycle sélectionné
    status = get_status_at_cycle(model, engine_data, feature_names, selected_cycle_idx)
    
    alert_level = status['alert_level']
    alert_color = t['alert_colors'][alert_level]
    alert_label = t['alert_labels'][alert_level]
    
    # Affichage des métriques
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric(t['current_cycle'], f"{status['current_cycle']} km (x1000)")
    
    with col_info2:
        st.metric(t['predicted_rul'], f"{status['predicted_rul']} {t['cycles']}")
    
    with col_info3:
        st.markdown(f"**{t['status']}**")
        st.markdown(
            f"<span style='background-color:{alert_color}; color:white; padding:8px 20px; border-radius:5px; font-weight:bold; display:inline-block; margin-top:5px;'>{alert_label}</span>",
            unsafe_allow_html=True
        )
    
    # Graphique RUL
    predictions = model.predict(engine_data[feature_names])
    cycles = engine_data['time_in_cycles'].values
    real_rul = engine_data['RUL'].values
    current_cycle_value = cycles[selected_cycle_idx]
    
    fig_rul = go.Figure()
    
    fig_rul.add_trace(go.Scatter(
        x=cycles, y=predictions,
        mode='lines', name=t['chart_rul_predicted'],
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_rul.add_trace(go.Scatter(
        x=cycles, y=real_rul,
        mode='lines', name=t['chart_rul_actual'],
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))
    
    fig_rul.add_hline(y=30, line_dash="dot", line_color="red",
                      annotation_text=t['chart_threshold'])
    
    fig_rul.add_vline(x=current_cycle_value, line_dash="solid", line_color="purple",
                      annotation_text=t['chart_current_cycle'])
    
    fig_rul.add_trace(go.Scatter(
        x=[current_cycle_value],
        y=[predictions[selected_cycle_idx]],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='circle'),
        name=t['chart_current_cycle'],
        showlegend=False
    ))
    
    fig_rul.update_layout(
        xaxis_title=t['xaxis_km'],
        yaxis_title=t['yaxis_cycles_remaining'],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50)
    )
    
    st.plotly_chart(fig_rul, use_container_width=True, config=STATIC_CONFIG)
    
    with st.expander(t['understand_chart']):
        st.markdown(t['chart_explanation_rul'])
    
    st.divider()
    
    # ----------------------------
    # SECTION 3 : PERFORMANCE
    # ----------------------------
    st.header(t['model_performance'])
    
    col_metrics, col_scatter = st.columns([1, 2])
    
    with col_metrics:
        st.subheader(t['metrics'])
        st.metric("MAE", "11.4 cycles")
        st.caption(t['mae_caption'])
        st.metric("RMSE", "16.4 cycles")
        st.caption(t['rmse_caption'])
        st.metric("R²", "0.84")
        st.caption(t['r2_caption'])
    
    with col_scatter:
        st.subheader(t['predictions_vs_reality'])
        
        y_true_sample, y_pred_sample = get_scatter_data(model, df, feature_names)
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_true_sample, y=y_pred_sample,
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='#1f77b4'),
            name=t['predictions']
        ))
        
        fig_scatter.add_trace(go.Scatter(
            x=[0, 125], y=[0, 125],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=t['perfect_prediction']
        ))
        
        fig_scatter.update_layout(
            xaxis_title=t['xaxis_rul_actual'],
            yaxis_title=t['yaxis_rul_predicted'],
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True, config=STATIC_CONFIG)
    
    with st.expander(t['understand_chart']):
        st.markdown(t['chart_explanation_scatter'])
    
    # Feature importance
    st.subheader(t['predictive_sensors'])
    
    importance = get_feature_importance(model, feature_names)
    top_features = importance.head(7).copy()
    top_features['label'] = top_features['feature'].map(t['sensor_labels']).fillna(top_features['feature'])
    
    fig_importance = go.Figure()
    
    fig_importance.add_trace(go.Bar(
        y=top_features['label'],
        x=top_features['importance'],
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig_importance.update_layout(
        xaxis_title=t['importance_xaxis'],
        yaxis_title="",
        height=300,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200)
    )
    
    st.plotly_chart(fig_importance, use_container_width=True, config=STATIC_CONFIG)
    
    with st.expander(t['understand_chart']):
        st.markdown(t['chart_explanation_importance'])
    
    st.divider()
    
    # ----------------------------
    # SECTION 4 : PIPELINE
    # ----------------------------
    st.header(t['pipeline'])
    
    cols = st.columns(5)
    for i, (title, items) in enumerate(t['pipeline_steps']):
        with cols[i]:
            st.markdown(f"**{title}**")
            for item in items:
                st.caption(item)
    
    st.divider()
    
    # ----------------------------
    # SECTION 5 : GLOSSAIRE
    # ----------------------------
    st.header(t['glossary_title'])
    
    col_glossary, col_stack = st.columns(2)
    
    with col_glossary:
        st.subheader(t['glossary'])
        st.markdown(t['glossary_table'])
    
    with col_stack:
        st.subheader(t['tech_stack'])
        st.markdown(t['tech_stack_content'])
    
    # ----------------------------
    # FOOTER
    # ----------------------------
    st.divider()
    st.caption("Jaliss 2025")


if __name__ == "__main__":
    main()