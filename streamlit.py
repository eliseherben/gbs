#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st


# In[ ]:


if "leeftijd" not in st.session_state:
    st.session_state.leeftijd = None
    
st.session_state._leeftijd = st.session_state.leeftijd

def set_leeftijd():
    st.session_state.leeftijd = st.session_state._leeftijd


# In[ ]:


if "geslacht" not in st.session_state:
    st.session_state.geslacht = None
    
st.session_state._geslacht = st.session_state.geslacht

def set_geslacht():
    st.session_state.geslacht = st.session_state._geslacht


# In[ ]:


if "eiwitgehalte" not in st.session_state:
    st.session_state.eiwitgehalte = None
    
st.session_state._eiwitgehalte = st.session_state.eiwitgehalte

def set_eiwitgehalte():
    st.session_state.eiwitgehalte = st.session_state._eiwitgehalte


# In[ ]:


if "spierkracht" not in st.session_state:
    st.session_state.spierkracht = None
    
st.session_state._spierkracht = st.session_state.spierkracht

def set_spierkracht():
    st.session_state.spierkracht = st.session_state._spierkracht


# In[ ]:


if "pijn" not in st.session_state:
    st.session_state.pijn = None
    
st.session_state._pijn = st.session_state.pijn

def set_pijn():
    st.session_state.pijn = st.session_state._pijn


# In[ ]:


if "infectie" not in st.session_state:
    st.session_state.infectie = None
    
st.session_state._infectie = st.session_state.infectie

def set_infectie():
    st.session_state.infectie = st.session_state._infectie


# In[ ]:


if "vaccinatie" not in st.session_state:
    st.session_state.vaccinatie = None
    
st.session_state._vaccinatie = st.session_state.vaccinatie

def set_vaccinatie():
    st.session_state.vaccinatie = st.session_state._vaccinatie


# In[ ]:


# Stap 1: Dataset genereren (zelfde als eerder)
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'leeftijd': np.random.randint(20, 80, size=n_samples),
    'geslacht': np.random.randint(0, 2, size=n_samples),  # 0 voor man, 1 voor vrouw
    'eiwitgehalte': np.random.uniform(5, 20, size=n_samples),
    'spierkracht': np.random.uniform(30, 100, size=n_samples),
    'pijn': np.random.randint(0, 2, size=n_samples),  # 0 voor geen pijn, 1 voor pijn
    'infectie': np.random.randint(0, 2, size=n_samples),  # 0 voor geen infectie, 1 voor infectie
    'vaccinatie': np.random.randint(0, 2, size=n_samples)  # 0 voor geen vaccinatie, 1 voor vaccinatie
})
data['doel'] = np.random.choice([0, 1, 2], size=n_samples)

# Dummy variabelen maken voor de categorische kenmerken 'geslacht', 'pijn', 'infectie' en 'vaccinatie'
data = pd.get_dummies(data, columns=['geslacht', 'pijn', 'infectie', 'vaccinatie'], drop_first=True)

# Splitsen in features (X) en target (y)
X = data[['leeftijd', 'geslacht_1', 'eiwitgehalte', 'spierkracht', 'pijn_1', 'infectie_1', 'vaccinatie_1']]
y = data['doel']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standaardiseren van kenmerken
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Stap 2: Multinomial Logistic Regression model trainen
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Streamlit applicatie
st.title('Voorspelling uitkomst GBS')

# Invoeropties voor de gebruiker met placeholders
leeftijd = st.number_input('Leeftijd', min_value=0, max_value=120, value=None, step=1, format="%d", key='_leeftijd', on_change = set_leeftijd, placeholder='Voer de leeftijd in')
geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_geslacht', on_change = set_geslacht, placeholder='Selecteer het geslacht')
eiwtgehalte = st.number_input('Eiwitgehalte bij binnenkomst', min_value=0.0, max_value=50.0, value=None, step=0.1, format="%.1f", key='_eiwitgehalte', on_change = set_eiwitgehalte, placeholder='Voer het eiwitgehalte bij binnenkomst in')
spierkracht = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_spierkracht', on_change = set_spierkracht, placeholder='Voer de spierkracht bij binnenkomst in')
pijn = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_pijn', on_change = set_pijn, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
infectie = st.selectbox('Heb je een voorgaande infectie gehad?', options=['Nee', 'Ja'], index=None, key='_infectie', on_change = set_infectie, placeholder = 'Selecteer of er een voorgaande infectie was')
vaccinatie = st.selectbox('Heb je een voorgaande vaccinatie gehad?', options=['Nee', 'Ja'], index=None, key='_vaccinatie', on_change = set_vaccinatie, placeholder = 'Selecteer of er een voorgaande vaccinatie was')

# Resultaten berekenen en tonen na het klikken op de knop
if st.button('Voorspelling maken'):
    if None in [leeftijd, eiwtgehalte, spierkracht, geslacht, pijn, infectie, vaccinatie]:
        st.error('Vul alle velden in om een voorspelling te maken.')
    else:
        # Omzetten van invoer naar modelinput
        nieuwe_observatie = pd.DataFrame({
            'leeftijd': [leeftijd],
            'geslacht_1': [1 if geslacht == 'Vrouw' else 0],
            'eiwitgehalte': [eiwtgehalte],
            'spierkracht': [spierkracht],
            'pijn_1': [1 if pijn == 'Ja' else 0],
            'infectie_1': [1 if infectie == 'Ja' else 0],
            'vaccinatie_1': [1 if vaccinatie == 'Ja' else 0]
        })

        # Standaardiseren van de nieuwe observatie
        nieuwe_observatie_scaled = scaler.transform(nieuwe_observatie)

        # Kansvoorspelling maken
        kansen = model.predict_proba(nieuwe_observatie_scaled)
        kans_resultaat = pd.DataFrame(kansen, columns=['niet lopen', 'lopen', 'dood'])

        # Resultaten tonen in Streamlit als metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{kans_resultaat['niet lopen'][0]:.2%}")
        col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{kans_resultaat['lopen'][0]:.2%}")
        col3.metric(label="Kans op overlijden", value=f"{kans_resultaat['dood'][0]:.2%}")

# Reset knop om alle invoeropties terug te zetten naar de placeholder waarden
if st.button('Nieuwe voorspelling maken'):
    st.session_state.clear()
    st.rerun()

