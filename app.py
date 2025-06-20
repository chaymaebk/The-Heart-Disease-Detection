import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Détection Cardiaque", layout="wide")

st.markdown("<h1 style='text-align: center; color: red;'> Détection de Maladie Cardiaque</h1>", unsafe_allow_html=True)
st.image("heart.jpg", width=150)

# Charger encodeurs, scaler et modèle sauvegardés
with open('encodeurs.pkl', 'rb') as f:
    encodeurs = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def preprocess_input(data, encodeurs, scaler):
    # Créer un dataframe avec une ligne
    df = pd.DataFrame([data])

    # Colonnes catégorielles à encoder
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    for col in categorical_cols:
        le = encodeurs[col]
        # Transformer la colonne avec le label encoder (attention : si la valeur n'existe pas dans l'encodeur, cela donnera une erreur)
        df[col] = le.transform(df[col])
    
    # Colonnes numériques à scaler
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# Interface utilisateur
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input('Âge', min_value=1, max_value=120, value=50)
    Sex = st.selectbox('Sexe', options=encodeurs['Sex'].classes_)
    ChestPainType = st.selectbox('Type de douleur thoracique', options=encodeurs['ChestPainType'].classes_)
    RestingBP = st.number_input('Pression artérielle au repos', min_value=50, max_value=250, value=120)
    Cholesterol = st.number_input('Cholestérol', min_value=100, max_value=600, value=200)

with col2:
    FastingBS = st.selectbox('Glycémie à jeun > 120 mg/dl ?', options=[0, 1])
    RestingECG = st.selectbox('ECG au repos', options=encodeurs['RestingECG'].classes_)
    MaxHR = st.number_input('Fréquence cardiaque maximale', min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox('Angine à l’effort ?', options=encodeurs['ExerciseAngina'].classes_)
    Oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    ST_Slope = st.selectbox('Pente ST', options=encodeurs['ST_Slope'].classes_)


if st.button('Prédire'):
    input_data = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }

    try:
        X_new = preprocess_input(input_data, encodeurs, scaler)
        pred = model.predict(X_new)[0]

        if pred == 1:
            st.error(" Résultat : Présence de maladie cardiaque.")
        else:
            st.success(" Résultat : Absence de maladie cardiaque.")
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")
