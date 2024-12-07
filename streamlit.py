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


cols = ['Age', 'Sex', 'MRC_sum_e', 'MRC_sum_w1', 'CNI_e', 'CNI_Bulbar_e',
       'CNI_Facial_e', 'CNI_Oculomotor_e', 'CNI_w1', 'CNI_Bulbar_w1',
       'CNI_Facial_w1', 'CNI_Oculomotor_w1', 'Pain_e', 'Pain_w1',
       'Sens_deficits_e', 'Sens_deficits_w1', 'Ant_event', 'Ant_event_GE',
       'Ant_event_respiratory', 'Ant_event_vaccination', 'Ant_event_other',
       'GBSDS_e', 'Able_to_walk_e', 'GBSDS_w1', 'Able_to_walk_w1',
       'CSF_protein_level', 'Comorbidity_respiration',
       'Days_between_onset_and_admission', 'Continent', 'Country_of_inclusion',
       'Surv_days']

for col in cols:
    if col not in st.session_state:
        # Initialize the session state variable dynamically
        setattr(st.session_state, col, None)

    # Set the corresponding _col variable dynamically
    setattr(st.session_state, f"_{col}", getattr(st.session_state, col))

    # Define the set function dynamically
    def set_col():
        setattr(st.session_state, col, getattr(st.session_state, f"_{col}"))


# In[ ]:


# for col in cols: 
#     if col not in st.session_state:
#         st.session_state.col = None

#     st.session_state._col = st.session_state.col

#     def set_col():
#         st.session_state.col = st.session_state._col


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


# age, mrc_sum_e, surv days, pain w1, comorbididty respiration, gbsds


# In[ ]:


# # Streamlit applicatie
# st.title('Voorspelling op de uitkomst van GBS')

# # Invoeropties voor de gebruiker met placeholders
# leeftijd = st.number_input('Leeftijd', min_value=0, max_value=120, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
# st.markdown(st.session_state.Age)

# geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
# st.markdown(st.session_state.Sex)

# spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0.0, max_value=60.0, value=None, step=1, key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
# st.markdown(st.session_state.MRC_sum_e)

# spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0.0, max_value=60.0, value=None, step=1, key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
# st.markdown(st.session_state.MRC_sum_w1)


# # spierkracht = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_spierkracht', on_change = set_spierkracht, placeholder='Voer de spierkracht bij binnenkomst in')
# # pijn = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_pijn', on_change = set_pijn, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
# # infectie = st.selectbox('Heb je een voorgaande infectie gehad?', options=['Nee', 'Ja'], index=None, key='_infectie', on_change = set_infectie, placeholder = 'Selecteer of er een voorgaande infectie was')
# # vaccinatie = st.selectbox('Heb je een voorgaande vaccinatie gehad?', options=['Nee', 'Ja'], index=None, key='_vaccinatie', on_change = set_vaccinatie, placeholder = 'Selecteer of er een voorgaande vaccinatie was')


# In[ ]:


# # Stap 1: Dataset genereren (zelfde als eerder)
# np.random.seed(42)
# n_samples = 200

# data = pd.DataFrame({
#     'leeftijd': np.random.randint(20, 80, size=n_samples),
#     'geslacht': np.random.randint(0, 2, size=n_samples),  # 0 voor man, 1 voor vrouw
#     'eiwitgehalte': np.random.uniform(5, 20, size=n_samples),
#     'spierkracht': np.random.uniform(30, 100, size=n_samples),
#     'pijn': np.random.randint(0, 2, size=n_samples),  # 0 voor geen pijn, 1 voor pijn
#     'infectie': np.random.randint(0, 2, size=n_samples),  # 0 voor geen infectie, 1 voor infectie
#     'vaccinatie': np.random.randint(0, 2, size=n_samples)  # 0 voor geen vaccinatie, 1 voor vaccinatie
# })
# data['doel'] = np.random.choice([0, 1, 2], size=n_samples)

# # Dummy variabelen maken voor de categorische kenmerken 'geslacht', 'pijn', 'infectie' en 'vaccinatie'
# data = pd.get_dummies(data, columns=['geslacht', 'pijn', 'infectie', 'vaccinatie'], drop_first=True)

# # Splitsen in features (X) en target (y)
# X = data[['leeftijd', 'geslacht_1', 'eiwitgehalte', 'spierkracht', 'pijn_1', 'infectie_1', 'vaccinatie_1']]
# y = data['doel']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Standaardiseren van kenmerken
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Stap 2: Multinomial Logistic Regression model trainen
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(X_train, y_train)

# # Streamlit applicatie
# st.title('Voorspelling uitkomst GBS')

# # Invoeropties voor de gebruiker met placeholders
# leeftijd = st.number_input('Leeftijd', min_value=0, max_value=120, value=None, step=1, format="%d", key='_leeftijd', on_change = set_leeftijd, placeholder='Voer de leeftijd in')
# geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_geslacht', on_change = set_geslacht, placeholder='Selecteer het geslacht')
# eiwtgehalte = st.number_input('Eiwitgehalte bij binnenkomst', min_value=0.0, max_value=50.0, value=None, step=0.1, format="%.1f", key='_eiwitgehalte', on_change = set_eiwitgehalte, placeholder='Voer het eiwitgehalte bij binnenkomst in')
# spierkracht = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_spierkracht', on_change = set_spierkracht, placeholder='Voer de spierkracht bij binnenkomst in')
# pijn = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_pijn', on_change = set_pijn, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
# infectie = st.selectbox('Heb je een voorgaande infectie gehad?', options=['Nee', 'Ja'], index=None, key='_infectie', on_change = set_infectie, placeholder = 'Selecteer of er een voorgaande infectie was')
# vaccinatie = st.selectbox('Heb je een voorgaande vaccinatie gehad?', options=['Nee', 'Ja'], index=None, key='_vaccinatie', on_change = set_vaccinatie, placeholder = 'Selecteer of er een voorgaande vaccinatie was')

# # Resultaten berekenen en tonen na het klikken op de knop
# if st.button('Voorspelling maken'):
#     if None in [leeftijd, eiwtgehalte, spierkracht, geslacht, pijn, infectie, vaccinatie]:
#         st.error('Vul alle velden in om een voorspelling te maken.')
#     else:
#         # Omzetten van invoer naar modelinput
#         nieuwe_observatie = pd.DataFrame({
#             'leeftijd': [leeftijd],
#             'geslacht_1': [1 if geslacht == 'Vrouw' else 0],
#             'eiwitgehalte': [eiwtgehalte],
#             'spierkracht': [spierkracht],
#             'pijn_1': [1 if pijn == 'Ja' else 0],
#             'infectie_1': [1 if infectie == 'Ja' else 0],
#             'vaccinatie_1': [1 if vaccinatie == 'Ja' else 0]
#         })

#         # Standaardiseren van de nieuwe observatie
#         nieuwe_observatie_scaled = scaler.transform(nieuwe_observatie)

#         # Kansvoorspelling maken
#         kansen = model.predict_proba(nieuwe_observatie_scaled)
#         kans_resultaat = pd.DataFrame(kansen, columns=['niet lopen', 'lopen', 'dood'])

#         # Resultaten tonen in Streamlit als metrics
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{kans_resultaat['niet lopen'][0]:.2%}")
#         col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{kans_resultaat['lopen'][0]:.2%}")
#         col3.metric(label="Kans op overlijden", value=f"{kans_resultaat['dood'][0]:.2%}")

# # Reset knop om alle invoeropties terug te zetten naar de placeholder waarden
# if st.button('Nieuwe voorspelling maken'):
#     st.session_state.clear()
#     st.rerun()


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
# st.pills("**Datapunten**", 
#                options = ["Alleen variabelen bij binnenkomst meenemen", 
#                           "Variabelen bij zowel binnenkomst als week 1 meenemen"], 
#                selection_mode="single", key='_datapunten', on_change = set_datapunten,
#               )


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


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pickle
from statsmodels.tools.tools import add_constant

st.markdown("### Invoermogelijkheden")
if st.session_state._modelkeuze == None or st.session_state._significantie == None:
    st.markdown("Vul de specificaties in")

    
# df1_2 = pd.read_csv("df_streamlit_1.2.csv", index_col=0)
# df2_1 = pd.read_csv("df_streamlit_2.1.csv", index_col=0)
# df2_2 = pd.read_csv("df_streamlit_2.2.csv", index_col=0)

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

# leeftijd = st.number_input('Leeftijd', min_value=2, max_value=100, value=None, step=1, format="%d", key='_Age', on_change = set_Age, placeholder='Voer de leeftijd in')
# geslacht = st.selectbox('Geslacht', options=['Man', 'Vrouw'], index=None, key='_Sex', on_change = set_Sex, placeholder='Selecteer het geslacht')
# spierkracht_e = st.number_input('Spierkracht bij binnenkomst', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_e', on_change = set_MRC_sum_e, placeholder='Voer de spierkracht bij binnenkomst in')
# spierkracht_w1 = st.number_input('Spierkracht bij week 1', min_value=0, max_value=60, value=None, step=1, format="%d", key='_MRC_sum_w1', on_change = set_MRC_sum_w1, placeholder='Voer de spierkracht bij week 1 in')
# pijn = st.selectbox('Pijn bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_pijn', on_change = set_pijn, placeholder = 'Selecteer of er sprake was van pijn bij binnenkomst')
# gevoelsstoornis_e = st.selectbox('Gevoelsstoornis bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_e', on_change = set_Sens_deficits_e, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij binnenkomst')
# gevoelsstoornis_w1 = st.selectbox('Gevoelsstoornis bij week 1', options=['Nee', 'Ja'], index=None, key='_Sens_deficits_w1', on_change = set_Sens_deficits_w1, placeholder = 'Selecteer of er sprake was van een gevoelsstoornis bij week 1')
# hersenzenuw = st.selectbox('Uitval/aantasting hersenzenuwen bij binnenkomst', options=['Nee', 'Ja'], index=None, key='_CNI_e', on_change = set_CNI_e, placeholder = 'Selecteer of er sprake was van uitval of aantasting van de hersenzenuwen bij binnenkomst')

# input_data = pd.DataFrame({
#     'Age': [leeftijd],
#     'MRC_sum_e': [spierkracht_e],
#     'MRC_sum_w1': [spierkracht_w1],
#     'Sens_deficits_e_2.0': [1 if gevoelsstoornis_e == "nvt" else 0],
#     'Sens_deficits_w1_1.0': [1 if gevoelsstoornis_w1 == "Ja" else 0],
#     'Sens_deficits_w1_2.0': [1 if gevoelsstoornis_w1 == "nvt" else 0],
#     'CNI_e_1.0': [1 if hersenzenuw == "Ja" else 0],
#     'Pain_e_1.0': [1 if pijn == "Ja" else 0]
# })

# if st.button('Voorspelling maken'):
#     if None in [leeftijd, spierkracht_e, spierkracht_w1, geslacht, gevoelsstoornis_e, 
#                 gevoelsstoornis_w1, hersenzenuw, pijn]:
#         st.error('Vul alle velden in om een voorspelling te maken.')
    
#     prediction_result = predict_outcome(df, input_data)
    
#     st.write("Kans op uitkomst:")
    
#     col1, col2, col3 = st.columns(3)
#     col1.metric(label="Kans op **niet** zelfstandig kunnen lopen", value=f"{prediction_result['niet lopen'][0]:.2%}")
#     col2.metric(label="Kans op zelfstandig kunnen lopen", value=f"{prediction_result['lopen'][0]:.2%}")
#     col3.metric(label="Kans op overlijden", value=f"{prediction_result['dood'][0]:.2%}")

# if st.button('Nieuwe voorspelling maken'):
#     st.session_state.clear()
#     st.rerun()


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




