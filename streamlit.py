#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st


# ### Session state

# In[ ]:


if "Age" not in st.session_state:
    st.session_state.Age = None
    
st.session_state._Age = st.session_state.Age

def set_Age():
    st.session_state.Age = st.session_state._Age


# In[ ]:


if "Sex" not in st.session_state:
    st.session_state.Sex = None
    
st.session_state._Sex = st.session_state.Sex

def set_Sex():
    st.session_state.Sex = st.session_state._Sex


# In[ ]:


if "MRC_sum_e" not in st.session_state:
    st.session_state.MRC_sum_e = None
    
st.session_state._MRC_sum_e = st.session_state.MRC_sum_e

def set_MRC_sum_e():
    st.session_state.MRC_sum_e = st.session_state._MRC_sum_e


# In[ ]:


if "MRC_sum_w1" not in st.session_state:
    st.session_state.MRC_sum_w1 = None
    
st.session_state._MRC_sum_w1 = st.session_state.MRC_sum_w1

def set_MRC_sum_w1():
    st.session_state.MRC_sum_w1 = st.session_state._MRC_sum_w1


# In[ ]:


if "CNI_e" not in st.session_state:
    st.session_state.CNI_e = None
    
st.session_state._CNI_e = st.session_state.CNI_e

def set_CNI_e():
    st.session_state.CNI_e = st.session_state._CNI_e


# In[ ]:


if "CNI_w1" not in st.session_state:
    st.session_state.CNI_w1 = None
    
st.session_state._CNI_w1 = st.session_state.CNI_w1

def set_CNI_w1():
    st.session_state.CNI_w1 = st.session_state._CNI_w1


# In[ ]:


if "CNI_Bulbar_e" not in st.session_state:
    st.session_state.CNI_Bulbar_e = None
    
st.session_state._CNI_Bulbar_e = st.session_state.CNI_Bulbar_e

def set_CNI_Bulbar_e():
    st.session_state.CNI_Bulbar_e = st.session_state._CNI_Bulbar_e


# In[ ]:


if "CNI_Bulbar_w1" not in st.session_state:
    st.session_state.CNI_Bulbar_w1 = None
    
st.session_state._CNI_Bulbar_w1 = st.session_state.CNI_Bulbar_w1

def set_CNI_Bulbar_w1():
    st.session_state.CNI_Bulbar_w1 = st.session_state._CNI_Bulbar_w1


# In[ ]:


if "CNI_Facial_e" not in st.session_state:
    st.session_state.CNI_Facial_e = None
    
st.session_state._CNI_Facial_e = st.session_state.CNI_Facial_e

def set_CNI_Facial_e():
    st.session_state.CNI_Facial_e = st.session_state._CNI_Facial_e


# In[ ]:


if "CNI_Facial_w1" not in st.session_state:
    st.session_state.CNI_Facial_w1 = None
    
st.session_state._CNI_Facial_w1 = st.session_state.CNI_Facial_w1

def set_CNI_Facial_w1():
    st.session_state.CNI_Facial_w1 = st.session_state._CNI_Facial_w1


# In[ ]:


if "CNI_Oculomotor_e" not in st.session_state:
    st.session_state.CNI_Oculomotor_e = None
    
st.session_state._CNI_Oculomotor_e = st.session_state.CNI_Oculomotor_e

def set_CNI_Oculomotor_e():
    st.session_state.CNI_Oculomotor_e = st.session_state._CNI_Oculomotor_e


# In[ ]:


if "CNI_Oculomotor_w1" not in st.session_state:
    st.session_state.CNI_Oculomotor_w1 = None
    
st.session_state._CNI_Oculomotor_w1 = st.session_state.CNI_Oculomotor_w1

def set_CNI_w1():
    st.session_state.CNI_Oculomotor_w1 = st.session_state._CNI_Oculomotor_w1


# In[ ]:


if "Pain_e" not in st.session_state:
    st.session_state.Pain_e = None
    
st.session_state._Pain_e = st.session_state.Pain_e

def set_Pain_e():
    st.session_state.Pain_e = st.session_state._Pain_e


# In[ ]:


if "Pain_w1" not in st.session_state:
    st.session_state.Pain_w1 = None
    
st.session_state._Pain_w1 = st.session_state.Pain_w1

def set_Pain_w1():
    st.session_state.Pain_w1 = st.session_state._Pain_w1


# In[ ]:


if "Sens_deficits_e" not in st.session_state:
    st.session_state.Sens_deficits_e = None
    
st.session_state._Sens_deficits_e = st.session_state.Sens_deficits_e

def set_Sens_deficits_e():
    st.session_state.Sens_deficits_e = st.session_state._Sens_deficits_e


# In[ ]:


if "Sens_deficits_w1" not in st.session_state:
    st.session_state.Sens_deficits_w1 = None
    
st.session_state._Sens_deficits_w1 = st.session_state.Sens_deficits_w1

def set_Sens_deficits_w1():
    st.session_state.Sens_deficits_w1 = st.session_state._Sens_deficits_w1


# In[ ]:


if "Ant_event" not in st.session_state:
    st.session_state.Ant_event = None
    
st.session_state._Ant_event = st.session_state.Ant_event

def set_Ant_event():
    st.session_state.Ant_event = st.session_state._Ant_event


# In[ ]:


if "Ant_event_respiratory" not in st.session_state:
    st.session_state.Ant_event_respiratory = None
    
st.session_state._Ant_event_respiratory = st.session_state.Ant_event_respiratory

def set_Ant_event_respiratory():
    st.session_state.Ant_event_respiratory = st.session_state._Ant_event_respiratory


# In[ ]:


if "Ant_event_GE" not in st.session_state:
    st.session_state.Ant_event_GE = None
    
st.session_state._Ant_event_GE = st.session_state.Ant_event_GE

def set_Ant_event_GE():
    st.session_state.Ant_event_GE = st.session_state._Ant_event_GE


# In[ ]:


if "Ant_event_vaccination" not in st.session_state:
    st.session_state.Ant_event_vaccination = None
    
st.session_state._Ant_event_vaccination = st.session_state.Ant_event_vaccination

def set_Ant_event_vaccination():
    st.session_state.Ant_event_vaccination = st.session_state._Ant_event_vaccination


# In[ ]:


if "Ant_event_other" not in st.session_state:
    st.session_state.Ant_event_other = None
    
st.session_state._Ant_event_other = st.session_state.Ant_event_other

def set_Ant_event_other():
    st.session_state.Ant_event_other = st.session_state._Ant_event_other


# In[ ]:


if "GBSDS_e" not in st.session_state:
    st.session_state.GBSDS_e = None
    
st.session_state._GBSDS_e = st.session_state.GBSDS_e

def set_GBSDS_e():
    st.session_state.GBSDS_e = st.session_state._GBSDS_e


# In[ ]:


if "GBSDS_w1" not in st.session_state:
    st.session_state.GBSDS_w1 = None
    
st.session_state._GBSDS_w1 = st.session_state.GBSDS_w1

def set_GBSDS_w1():
    st.session_state.GBSDS_w1 = st.session_state._GBSDS_w1


# In[ ]:


if "Able_to_walk_e" not in st.session_state:
    st.session_state.Able_to_walk_e = None
    
st.session_state._Able_to_walk_e = st.session_state.Able_to_walk_e

def set_Able_to_walk_e():
    st.session_state.Able_to_walk_e = st.session_state._Able_to_walk_e


# In[ ]:


if "Able_to_walk_w1" not in st.session_state:
    st.session_state.Able_to_walk_w1 = None
    
st.session_state._Able_to_walk_w1 = st.session_state.Able_to_walk_w1

def set_Able_to_walk_w1():
    st.session_state.Able_to_walk_w1 = st.session_state._Able_to_walk_w1


# In[ ]:


if "CSF_protein_level" not in st.session_state:
    st.session_state.CSF_protein_level = None
    
st.session_state._CSF_protein_level = st.session_state.CSF_protein_level

def set_CSF_protein_level():
    st.session_state.CSF_protein_level = st.session_state._CSF_protein_level


# In[ ]:


if "Comorbidity_respiration" not in st.session_state:
    st.session_state.Comorbidity_respiration = None
    
st.session_state._Comorbidity_respiration = st.session_state.Comorbidity_respiration

def set_Comorbidity_respiration():
    st.session_state.Comorbidity_respiration = st.session_state._Comorbidity_respiration


# In[ ]:


if "Days_between_onset_and_admission" not in st.session_state:
    st.session_state.Days_between_onset_and_admission = None
    
st.session_state._Days_between_onset_and_admission = st.session_state.Days_between_onset_and_admission

def set_Days_between_onset_and_admission():
    st.session_state.Days_between_onset_and_admission = st.session_state._Days_between_onset_and_admission


# In[ ]:


if "Continent" not in st.session_state:
    st.session_state.Continent = None
    
st.session_state._Continent = st.session_state.Continent

def set_Continent():
    st.session_state.Continent = st.session_state._Continent


# In[ ]:


if "Country_of_inclusion" not in st.session_state:
    st.session_state.Country_of_inclusion = None
    
st.session_state._Country_of_inclusion = st.session_state.Country_of_inclusion

def set_Country_of_inclusion():
    st.session_state.Country_of_inclusion = st.session_state._Country_of_inclusion


# In[ ]:


if "Surv_days" not in st.session_state:
    st.session_state.Surv_days = None
    
st.session_state._Surv_days = st.session_state.Surv_days

def set_Surv_days():
    st.session_state.Surv_days = st.session_state._Surv_days


# In[ ]:


if "modelkeuze" not in st.session_state:
    st.session_state.modelkeuze = None
    
st.session_state._modelkeuze = st.session_state.modelkeuze

def set_modelkeuze():
    st.session_state.modelkeuze = st.session_state._modelkeuze


# In[ ]:


if "significantie" not in st.session_state:
    st.session_state.significantie = None
    
st.session_state._significantie = st.session_state.significantie

def set_significantie():
    st.session_state.significantie = st.session_state._significantie


# In[ ]:


if "datapunten" not in st.session_state:
    st.session_state.datapunten = None
    
st.session_state._datapunten = st.session_state.datapunten

def set_datapunten():
    st.session_state.datapunten = st.session_state._datapunten


# In[ ]:


st.title('Voorspelling uitkomst GBS')
st.markdown("De prognose van een patiënt kan gebasseerd worden op verschillende modellen en instellingen. Hierom worden er hieronder een aantal vragen gesteld, op basis van deze vragen zullen de invoermogelijkheden kunnen veranderen.")

st.markdown("### Specificatie")
st.pills("**Modelkeuze**", 
               options = ["Model 1: basismodel", "Model 2: interactietermen"], 
               selection_mode="single", key='_modelkeuze', on_change = set_modelkeuze,
              )
st.pills("**Significantie**", 
               options = ["Variabelen die significant zijn voor beide uitkomsten", 
                          "Variabelen die significant zijn voor minimaal één uitkomst"], 
               selection_mode="single", key='_significantie', on_change = set_significantie,
              )


# In[ ]:


def predict_outcome(df, X_input):
    X = df.drop(columns=['Uitkomst'])
    y = df['Uitkomst']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_const = add_constant(X_scaled) 
    
    y = y.astype('category')
    y = y.cat.reorder_categories([1, 0, 2], ordered=True)

    model = sm.MNLogit(y, X_const)
    result = model.fit(method='lbfgs')
    
    X_input_scaled = scaler.transform(X_input)
    
    X_input_const = add_constant(X_input_scaled, has_constant='add') 
    
    chances = result.predict(X_input_const)
    chance_df = pd.DataFrame(chances, columns=['lopen','niet lopen', 'dood'])
    return chance_df


# ### Model 1.1

# In[ ]:


if st.session_state._modelkeuze == "Model 1: basismodel" and st.session_state._significantie == "Variabelen die significant zijn voor beide uitkomsten":
    df = pd.read_csv("df_streamlit_1.1.csv", index_col=0)

    leeftijd = st.number_input('Leeftijd', min_value=2, max_value=100, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
    geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
    spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
    spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
    GBSDS_w1 = st.selectbox('GBS disability score bij week 1', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_w1', on_change = set_GBSDS_w1, placeholder = 'Selecteer wat de GBS disability score bij week 1 was')
    gevoelsstoornis_e = st.selectbox('Gevoelsstoornis bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_e', on_change = set_Sens_deficits_e, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij binnenkomst')
    gevoelsstoornis_w1 = st.selectbox('Gevoelsstoornis bij week 1', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_w1', on_change = set_Sens_deficits_w1, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij week 1')

    input_data = pd.DataFrame({
        'Age': [leeftijd],
        'MRC_sum_e': [spierkracht_e],
        'MRC_sum_w1': [spierkracht_w1],
        'Sens_deficits_e_2.0': [1 if gevoelsstoornis_e == "nvt" else 0],
        'Sens_deficits_w1_1.0': [1 if gevoelsstoornis_w1 == "Ja" else 0],
        'Sens_deficits_w1_2.0': [1 if gevoelsstoornis_w1 == "nvt" else 0],
        'GBSDS_w1_7.0': [1 if GBSDS_w1 == "7: Onbekend" else 0]
    })
 
    if st.button('Voorspelling maken'):
        if None in [leeftijd, spierkracht_e, spierkracht_w1, geslacht, gevoelsstoornis_e, 
                    gevoelsstoornis_w1, GBSDS_w1]:
            st.error('Vul alle velden in om een voorspelling te maken.')

        prediction_result = predict_outcome(df, input_data)

        st.write("Kans op uitkomst:")

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{prediction_result['niet lopen'][0]:.2%}")
        col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{prediction_result['lopen'][0]:.2%}")
        col3.metric(label="Kans op overlijden", value=f"{prediction_result['dood'][0]:.2%}")

    if st.button('Nieuwe voorspelling maken'):
        st.session_state.clear()
        st.rerun()


# ### Model 1.2

# In[ ]:


if st.session_state._modelkeuze == "Model 1: basismodel" and st.session_state._significantie == "Variabelen die significant zijn voor minimaal één uitkomst":
    df = pd.read_csv("df_streamlit_1.2.csv", index_col=0)

    leeftijd = st.number_input('Leeftijd', min_value=2, max_value=100, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
    geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
    spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
    spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
    GBSDS_w1 = st.selectbox('GBS disability score bij week 1', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_w1', on_change = set_GBSDS_w1, placeholder = 'Selecteer wat de GBS disability score bij week 1 was')
    GBSDS_e = st.selectbox('GBS disability score bij binnenkomst', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_e', on_change = set_GBSDS_e, placeholder = 'Selecteer wat de GBS disability score bij binnenkomst was')
    gevoelsstoornis_e = st.selectbox('Gevoelsstoornis bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_e', on_change = set_Sens_deficits_e, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij binnenkomst')
    gevoelsstoornis_w1 = st.selectbox('Gevoelsstoornis bij week 1', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_w1', on_change = set_Sens_deficits_w1, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij week 1')
    cni_e = st.selectbox('Uitval/aantasting hersenzenuwen bij binnenkomst', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_e', on_change = set_CNI_e, placeholder = 'Selecteer of er sprake was van uitval/aantasting van één of meer hersenzenuwen bij binnenkomst')
    cni_facial_e = st.selectbox('Zwakte van aangezichtsspieren bij binnenkomst', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Facial_e', on_change = set_CNI_Facial_e, placeholder = 'Selecteer of er sprake was van zwakte van aangezichtsspieren bij binnenkomst')
    cni_facial_w1 = st.selectbox('Zwakte van aangezichtsspieren bij week 1', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Facial_w1', on_change = set_CNI_Facial_w1, placeholder = 'Selecteer of er sprake was van zwakte van aangezichtsspieren bij week 1')
    cni_oculomotor_e = st.selectbox('Zwakte van oogbolspieren bij binnenkomst', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Oculomotor_e', on_change = set_CNI_Oculomotor_e, placeholder = 'Selecteer of er sprake was van zwakte van oogbolspieren bij binnenkomst')
    pain_e = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Pain_e', on_change = set_Pain_e, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
    comporbidity_respiration = st.selectbox('Luchtwegklachten', options=['Nee', 'Ja'], index=None, key='_Comorbidity_respiration', on_change = set_Comorbidity_respiration, placeholder = 'Selecteer of er sprake was van luchtwegklachten zoals bijv astma of COPD')
    country = st.selectbox('Land waar patiënt geïncludeerd is', options=['Nederland', 'Verenigd Koninkrijk', 'Duitsland', 'België', 'Denemarken', 
                                                                         'Italië', 'Spanje', 'Frankrijk', 'Griekenland', 'Zwitserland', 
                                                                         'Verenigde Staten', 'Canada', 'Argentinië', 'Brazilië', 'Japan', 
                                                                         'Taiwan', 'Maleisië', 'China', 'Bangladesh', 'Australië', 'Zuid-Afrika'], index=None, key='_Country_of_inclusion', on_change = set_Country_of_inclusion, placeholder = 'Selecteer het land waarin de patiënt geïncludeerd is')
    buikgriep = st.selectbox('Voorgaande infectie: buikgriep', options=['Nee', 'Ja'], index=None, key='_Ant_event_GE', on_change = set_Ant_event_GE, placeholder = 'Selecteer of er sprake was van een voorgaande buikgriep')


    input_data = pd.DataFrame({
        'Age': [leeftijd],
        'MRC_sum_e': [spierkracht_e],
        'MRC_sum_w1': [spierkracht_w1],
        'Ant_event_GE': [1 if buikgriep == "Ja" else 0],
        'CNI_e_1.0': [1 if cni_e == "Ja" else 0],
        'CNI_Facial_e_1.0': [1 if cni_facial_e == "Ja" else 0],
        'CNI_Oculomotor_e_1.0' : [1 if cni_oculomotor_e == "Ja" else 0],
        'CNI_Facial_w1_1.0' : [1 if cni_facial_w1 == "Ja" else 0],
        'Pain_e_1.0' : [1 if pain_e == "Ja" else 0],
        'Sens_deficits_e_2.0': [1 if gevoelsstoornis_e == "nvt" else 0],
        'Sens_deficits_w1_1.0': [1 if gevoelsstoornis_w1 == "Ja" else 0],
        'Sens_deficits_w1_2.0': [1 if gevoelsstoornis_w1 == "nvt" else 0],
        'Comorbidity_respiration_1.0' : [1 if comporbidity_respiration == "Ja" else 0],
        'GBSDS_e_2.0': [1 if GBSDS_e == "2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen" else 0],
        'GBSDS_e_5.0': [1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0],
        'GBSDS_w1_2.0': [1 if GBSDS_w1 == "2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen" else 0], 
        'Country_of_inclusion_4': [1 if country == "België" else 0], 
    })
 
    if st.button('Voorspelling maken'):
        if None in [leeftijd, spierkracht_e, spierkracht_w1, geslacht, buikgriep, cni_e, cni_facial_e, cni_oculomotor_e, 
                    cni_facial_w1, pain_e, gevoelsstoornis_e, gevoelsstoornis_w1, comporbidity_respiration, GBSDS_e, GBSDS_w1, country]:
            st.error('Vul alle velden in om een voorspelling te maken.')

        prediction_result = predict_outcome(df, input_data)

        st.write("Kans op uitkomst:")

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{prediction_result['niet lopen'][0]:.2%}")
        col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{prediction_result['lopen'][0]:.2%}")
        col3.metric(label="Kans op overlijden", value=f"{prediction_result['dood'][0]:.2%}")

    if st.button('Nieuwe voorspelling maken'):
        st.session_state.clear()
        st.rerun()


# ### Model 2.1

# In[ ]:


if st.session_state._modelkeuze == "Model 2: interactietermen" and st.session_state._significantie == "Variabelen die significant zijn voor beide uitkomsten":
    df = pd.read_csv("df_streamlit_2.1.csv", index_col=0)

    leeftijd = st.number_input('Leeftijd', min_value=2, max_value=100, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
    geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
    spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
    spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
    GBSDS_e = st.selectbox('GBS disability score bij binnenkomst', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_e', on_change = set_GBSDS_e, placeholder = 'Selecteer wat de GBS disability score bij binnenkomst was')
    gevoelsstoornis_e = st.selectbox('Gevoelsstoornis bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_e', on_change = set_Sens_deficits_e, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij binnenkomst')
    gevoelsstoornis_w1 = st.selectbox('Gevoelsstoornis bij week 1', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_w1', on_change = set_Sens_deficits_w1, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij week 1')
    cni_w1 = st.selectbox('Uitval/aantasting hersenzenuwen bij week 1', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_w1', on_change = set_CNI_w1, placeholder = 'Selecteer of er sprake was van uitval/aantasting van één of meer hersenzenuwen bij week 1')

    input_data = pd.DataFrame({
        'Age': [leeftijd],
        'MRC_sum_e': [spierkracht_e],
        'MRC_sum_w1': [spierkracht_w1],
        'Sens_deficits_e_1.0': [1 if gevoelsstoornis_e == "Ja" else 0],
        'MRC_sum_e_GBSDS_e_5.0_interaction': [(spierkracht_e if spierkracht_e != None else 0) * (1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)],
        'CNI_w1_2.0_GBSDS_e_4.0_interaction': [(1 if cni_w1 == "Niet mogelijk te bepalen" else 0) * (1 if GBSDS_e == "4: bedlegerig of stoelgebonden" else 0)],
        'Sens_deficits_w1_1.0_GBSDS_e_5.0_interaction': [(1 if gevoelsstoornis_w1 == "Ja" else 0) * (1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)]
    })
 
    if st.button('Voorspelling maken'):
        if None in [leeftijd, spierkracht_e, spierkracht_w1, geslacht, gevoelsstoornis_e, GBSDS_e, cni_w1, gevoelsstoornis_w1]:
            st.error('Vul alle velden in om een voorspelling te maken.')

        prediction_result = predict_outcome(df, input_data)

        st.write("Kans op uitkomst:")

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{prediction_result['niet lopen'][0]:.2%}")
        col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{prediction_result['lopen'][0]:.2%}")
        col3.metric(label="Kans op overlijden", value=f"{prediction_result['dood'][0]:.2%}")

    if st.button('Nieuwe voorspelling maken'):
        st.session_state.clear()
        st.rerun()


# ### Model 2.2

# In[ ]:


if st.session_state._modelkeuze == "Model 2: interactietermen" and st.session_state._significantie == "Variabelen die significant zijn voor minimaal één uitkomst":
    df = pd.read_csv("df_streamlit_2.2.csv", index_col=0)

    leeftijd = st.number_input('Leeftijd', min_value=2, max_value=100, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
    geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
    spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
    spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
    cni_w1 = st.selectbox('Uitval/aantasting hersenzenuwen bij week 1', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_w1', on_change = set_CNI_w1, placeholder = 'Selecteer of er sprake was van uitval/aantasting van één of meer hersenzenuwen bij week 1')
    cni_bulbar_e = st.selectbox('Spraak- en slikstoornissen bij binnenkomst', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Bulbar_e', on_change = set_CNI_Bulbar_e, placeholder = 'Selecteer of er sprake was van spraak- en slikstoornissen bij binnenkomst')
    cni_bulbar_w1 = st.selectbox('Spraak- en slikstoornissen bij week 1', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Bulbar_w1', on_change = set_CNI_Bulbar_w1, placeholder = 'Selecteer of er sprake was van spraak- en slikstoornissen bij week 1')
    cni_oculomotor_e = st.selectbox('Zwakte van oogbolspieren bij binnenkomst', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Oculomotor_e', on_change = set_CNI_Oculomotor_e, placeholder = 'Selecteer of er sprake was van zwakte van oogbolspieren bij binnenkomst')
    cni_facial_w1 = st.selectbox('Zwakte van aangezichtsspieren bij week 1', options=['Nee', 'Ja', 'Niet mogelijk te bepalen'], index=None, key='_CNI_Facial_w1', on_change = set_CNI_Facial_w1, placeholder = 'Selecteer of er sprake was van zwakte van aangezichtsspieren bij week 1')
    pain_e = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Pain_e', on_change = set_Pain_e, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
    pain_w1 = st.selectbox('Pijn bij week 1', options=['Nee', 'Ja'], index=None, key='_Pain_w1', on_change = set_Pain_w1, placeholder = 'Selecteer of er sprake was van pijn bij week 1')
    GBSDS_e = st.selectbox('GBS disability score bij binnenkomst', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_e', on_change = set_GBSDS_e, placeholder = 'Selecteer wat de GBS disability score bij binnenkomst was')
    GBSDS_w1 = st.selectbox('GBS disability score bij week 1', options=['0: gezond', '1: weinig symptomen en in staat om te rennen', '2: in staat om minimaal 10 meter te lopen zonder hulp, maar niet in staat om te rennen', 
                                                                     '3: in staat om 10 meter te lopen in een open ruimte met hulp', '4: bedlegerig of stoelgebonden', '5: ondersteuning nodig bij ademen, minimaal een deel van de dag', 
                                                                     '6: overleden', '7: onbekend'], index=None, key='_GBSDS_w1', on_change = set_GBSDS_w1, placeholder = 'Selecteer wat de GBS disability score bij week 1 was')
    gevoelsstoornis_e = st.selectbox('Gevoelsstoornis bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_e', on_change = set_Sens_deficits_e, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij binnenkomst')
    gevoelsstoornis_w1 = st.selectbox('Gevoelsstoornis bij week 1', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_w1', on_change = set_Sens_deficits_w1, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij week 1')
    lopen_e = st.selectbox('Vermogen om te lopen bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Able_to_walk_e', on_change = set_Able_to_walk_e, placeholder = 'Selecteer of er de patiënt kon lopen bij binnenkomst')
    lopen_w1 = st.selectbox('Vermogen om te lopen bij week 1', options=['Nee', 'Ja'], index=None, key='_Able_to_walk_w1', on_change = set_Able_to_walk_w1, placeholder = 'Selecteer of er de patiënt kon lopen bij week 1')
    comporbidity_respiration = st.selectbox('Luchtwegklachten', options=['Nee', 'Ja'], index=None, key='_Comorbidity_respiration', on_change = set_Comorbidity_respiration, placeholder = 'Selecteer of er sprake was van luchtwegklachten zoals bijv astma of COPD')
    infectie = st.selectbox('Voorgaande infectie', options=['Nee', 'Ja'], index=None, key='_Ant_event', on_change = set_Ant_event, placeholder = 'Selecteer of er sprake was van een voorgaande infectie')
    protein_level = st.number_input('Eiwitgehalte in hersenvocht', value=None, key='_CSF_protein_level', on_change = set_CSF_protein_level, placeholder='Voer de eiwitgehalte in hersenvocht in g/L in')
    country = st.selectbox('Land waar patiënt geïncludeerd is', options=['Nederland', 'Verenigd Koninkrijk', 'Duitsland', 'België', 'Denemarken', 
                                                                         'Italië', 'Spanje', 'Frankrijk', 'Griekenland', 'Zwitserland', 
                                                                         'Verenigde Staten', 'Canada', 'Argentinië', 'Brazilië', 'Japan', 
                                                                         'Taiwan', 'Maleisië', 'China', 'Bangladesh', 'Australië', 'Zuid-Afrika'], index=None, key='_Country_of_inclusion', on_change = set_Country_of_inclusion, placeholder = 'Selecteer het land waarin de patiënt geïncludeerd is')
    
    input_data = pd.DataFrame({
        'Age': [leeftijd],
        'MRC_sum_e': [spierkracht_e],
        'Ant_event': [1 if infectie == "Ja" else 0],
        'Sens_deficits_e_1.0': [1 if gevoelsstoornis_e == "Ja" else 0],
        'Comorbidity_respiration_1.0' : [1 if comporbidity_respiration == "Ja" else 0],
        'GBSDS_e_3.0': [1 if GBSDS_e == "3: in staat om 10 meter te lopen in een open ruimte met hulp" else 0],
        'GBSDS_w1_3.0': [1 if GBSDS_w1 == "3: in staat om 10 meter te lopen in een open ruimte met hulp" else 0],
        'Country_of_inclusion_4': [1 if country == "België" else 0], 
        'Sex_Able_to_walk_w1_1.0_interaction': [(1 if geslacht == "Man" else 0) * (1 if lopen_e == "Ja" else 0)], 
        'Sex_GBSDS_e_4.0_interaction': [(1 if geslacht == "Man" else 0) * (1 if GBSDS_e == "4: bedlegerig of stoelgebonden" else 0)], 
        'MRC_sum_e_GBSDS_e_5.0_interaction': [(spierkracht_e if spierkracht_e != None else 0) * (1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)], 
        'MRC_sum_w1_GBSDS_w1_5.0_interaction': [(spierkracht_w1 if spierkracht_w1 != None else 0) * (1 if GBSDS_w1 == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)],
        'CSF_protein_level_Country_of_inclusion_13_interaction': [(protein_level if protein_level != None else 0) * (1 if country == "Argentinië" else 0)],
        'CNI_Bulbar_e_1.0_GBSDS_e_4.0_interaction': [(1 if cni_bulbar_e == "Ja" else 0) * (1 if GBSDS_e == "4: bedlegerig of stoelgebonden" else 0)], 
        'CNI_Oculomotor_e_1.0_Pain_w1_1.0_interaction': [(1 if cni_oculomotor_e == "Ja" else 0) * (1 if pain_w1 == "Ja" else 0)],
        'CNI_w1_1.0_Sens_deficits_w1_1.0_interaction': [(1 if cni_w1 == "Ja" else 0) * (1 if gevoelsstoornis_w1 == "Ja" else 0)],
        'CNI_w1_1.0_Sens_deficits_w1_2.0_interaction': [(1 if cni_w1 == "Ja" else 0) * (1 if gevoelsstoornis_w1 == "nvt" else 0)],
        'CNI_w1_1.0_GBSDS_w1_4.0_interaction': [(1 if cni_w1 == "Ja" else 0) * (1 if GBSDS_w1 == "4: bedlegerig of stoelgebonden" else 0)],
        'CNI_w1_2.0_Pain_e_1.0_interaction': [(1 if cni_w1 == "Niet mogelijk te bepalen" else 0) * (1 if pain_e == "Ja" else 0)],
        'CNI_w1_2.0_GBSDS_e_4.0_interaction': [(1 if cni_w1 == "Niet mogelijk te bepalen" else 0) * (1 if GBSDS_e == "4: bedlegerig of stoelgebonden" else 0)],
        'CNI_Bulbar_w1_1.0_Pain_w1_1.0_interaction': [(1 if cni_bulbar_w1 == "Ja" else 0) * (1 if pain_e == "Ja" else 0)],
        'CNI_Facial_w1_1.0_GBSDS_e_4.0_interaction': [(1 if cni_facial_w1 == "Ja" else 0) * (1 if GBSDS_e == "4: bedlegerig of stoelgebonden" else 0)],
        'Pain_w1_1.0_Sens_deficits_w1_2.0_interaction': [(1 if pain_w1 == "Ja" else 0) * (1 if gevoelsstoornis_w1 == "nvt" else 0)],
        'Pain_w1_1.0_Able_to_walk_w1_1.0_interaction': [(1 if pain_w1 == "Ja" else 0) * (1 if lopen_w1 == "Ja" else 0)],
        'Sens_deficits_e_2.0_GBSDS_e_5.0_interaction': [(1 if gevoelsstoornis_e == "nvt" else 0) * (1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)],
        'Sens_deficits_w1_1.0_GBSDS_e_5.0_interaction': [(1 if gevoelsstoornis_w1 == "Ja" else 0) * (1 if GBSDS_e == "5: ondersteuning nodig bij ademen, minimaal een deel van de dag" else 0)],
        'Sens_deficits_w1_2.0_Country_of_inclusion_6_interaction': [(1 if gevoelsstoornis_w1 == "nvt" else 0) * (1 if country == "Italië" else 0)],
        'Able_to_walk_e_1.0_GBSDS_w1_4.0_interaction': [(1 if lopen_e == "Ja" else 0) * (1 if GBSDS_w1 == "4: bedlegerig of stoelgebonden" else 0)],
    })
 
    if st.button('Voorspelling maken'):
        if None in [leeftijd, geslacht, spierkracht_e, spierkracht_w1, cni_w1, cni_bulbar_e, cni_bulbar_w1, cni_oculomotor_e, 
        cni_facial_w1, pain_e, pain_w1, GBSDS_e, GBSDS_w1, gevoelsstoornis_e, gevoelsstoornis_w1, lopen_e, lopen_w1, 
        comporbidity_respiration, infectie, protein_level, country]:
            st.error('Vul alle velden in om een voorspelling te maken.')

        prediction_result = predict_outcome(df, input_data)

        st.write("Kans op uitkomst:")

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{prediction_result['niet lopen'][0]:.2%}")
        col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{prediction_result['lopen'][0]:.2%}")
        col3.metric(label="Kans op overlijden", value=f"{prediction_result['dood'][0]:.2%}")

    if st.button('Nieuwe voorspelling maken'):
        st.session_state.clear()
        st.rerun()


# In[ ]:




