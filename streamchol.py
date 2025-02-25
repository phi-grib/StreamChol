# Copyright 2025 Pablo Rodríguez Belenguer  
#  
# This file is part of StreamChol  
#  
# StreamChol is free software: you can redistribute it and/or modify  
# it under the terms of the GPL v3  
#  
# The homepage design of StreamChol was inspired by SerotoninAI (Łapińska et al., 2023).  
# Specifically, the code from lines 72 to 120. We acknowledge its influence in guiding  
# our initial interface layout, while all functionalities, objectives, and workflows  
# of StreamChol were developed separately.  


from turtle import onclick
import streamlit as st
import base64
import numpy as np
import pandas as pd
from streamlit_ketcher import st_ketcher
from PIL import Image
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
import pyarrow as pa
import contextvars
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.neighbors import KNeighborsRegressor
import random
# from chembl_webresource_client.new_client import new_client
# import seaborn as sns
import matplotlib.pyplot as plt
import uuid
from itertools import combinations
from rdkit.Chem import Draw
from plotnine import *
from PIL import Image
from io import BytesIO
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_validate, KFold
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef, accuracy_score, auc,make_scorer, recall_score
import time
from streamlit_option_menu import option_menu
import io
import pickle
# from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import os
import subprocess

# os.environ["R_HOME"] = 'C:/Program Files/R/R-4.3.1' 
# os.environ["PATH"] = 'C:/Program Files/R/R-4.3.1/bin/x64' + ";" + os.environ["PATH"] 

import rpy2.rinterface
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

pandas2ri.activate()

st.set_page_config(page_title="StreamChol")

image_path = "drawi3.svg"
image = open(image_path, "rb").read()

# Set the background image for the application
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/svg+xml;base64,{base64.b64encode(image).decode()}");
background-size: cover;
background-position: top;
#background-repeat: repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);

}}
</style>
"""

st.markdown("<h1 style='text-align: center; fontSize: 100px; font-style: italic; color: grey;'>StreamChol</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; fontSize: 40px; font-style: italic; color: grey;'>Prediction of Cholestasis for drug-like compounds</h1>", unsafe_allow_html=True)

#footer
footer_style = """
    position: fixed;
    left: 0;
    z-index: 3;
    bottom: 0;
    width: 100%;
    color: white;
    font-style: italic;
    text-align: left;
    padding: 10px;
    font-size: 16px;
"""
st.markdown(
    f'<div style="{footer_style}">Copyright (C) 2025 Pablo Rodríguez Belenguer</div>',
    unsafe_allow_html=True
)

if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 2
    manual_select = st.session_state['menu_option']
else:
    manual_select = None

selected = option_menu(None, ["Home", "Tutorial", "Data", "PK analysis", "Prediction","Report","Contact"],
                        icons=['house', "book-fill",'upload', "graph-up",  "calculator","book","envelope-at-fill"],
                        orientation="horizontal", manual_select=manual_select, key='menu_20', menu_icon='cast',default_index = 0,
                        styles={
        "container": {"padding": "21!important", "background-color":"#b4bbbf", "width": "auto"},
        "icon": {"color": "#4e5152", "font-size": "25px", "text-align" : "center"}, 
        "nav-link": {"font-size": "25px", "text-align": "center", "margin":"15px", "--hover-color": "#757473", "font-color":"#0a0a0a"},
        "nav-link-selected": {"background-color": "#2E6E88"},
        })


@st.cache_data
def smi_to_png(smi: str) -> str:
    """Returns molecular image as data URI."""
    mol = rdkit.Chem.MolFromSmiles(smi)
    pil_image = rdkit.Chem.Draw.MolToImage(mol)

    with io.BytesIO() as buffer:
        pil_image.save(buffer, "png")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{data}"

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def especificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    esp = especificity(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Define the metrics text with HTML and inline styles
    st.markdown("<h2 style='text-align: left; color: #696969; font-size: 23px'>Metrics</h2>", unsafe_allow_html=True)

    metrics_text = f"""
    <style>
    .custom-text {{
        font-size: 20px;
        color: #696969;
    }}
    </style>
    <p class="custom-text">
        - Accuracy: <strong style="color:#696969;">{acc:.2f}</strong>
        <br>
        - Sensitivity: <strong style="color:#696969;">{sens:.2f}</strong>
        <br>
        - Specificity: <strong style="color:#696969;">{esp:.2f}</strong>
        <br>
        - ROC AUC Score: <strong style="color:#696969;">{roc_auc:.2f}</strong>
        <br>
        - MCC: <strong style="color:#696969;">{mcc:.2f}</strong>
    </p>
    """

    # Display the metrics using st.markdown
    st.markdown(metrics_text, unsafe_allow_html=True)

def load(nombre=''):
    
    with open(f"model_{nombre}.pkl", "rb") as f:
        opciones = pickle.load(f)
    return opciones

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
def py2rpy_pandasdataframe(obj):
    from collections import OrderedDict
    od = OrderedDict()
    for name, values in obj.items():  # Cambio de iteritems() a items()
        try:
            od[name] = pandas2ri.py2rpy(values)
        except Exception as e:
            print('Error while trying to convert the column "%s". Fall back to string conversion. The error is: %s' % (name, str(e)))
            od[name] = ro.StrVector(values)
    return ro.DataFrame(od)

def button(*args, key=None, **kwargs):
    """
    Works just like a normal streamlit button, but it remembers its state, so that
    it works as a toggle button. If you click it, it will be pressed, and if you click
    it again, it will be unpressed. Args and output are the same as for
    [st.button](https://docs.streamlit.io/library/api-reference/widgets/st.button)
    """

    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if "type" not in kwargs:
        kwargs["type"] = "primary" if st.session_state[key] else "secondary"

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]
        st.experimental_rerun()

    return st.session_state[key]

def display_current_graph(on):
    graph1 = st.session_state.df_graph1[st.session_state.count]
    graph2 = st.session_state.df_graph2[st.session_state.count]
    stats = st.session_state.tkstats[st.session_state.count]

    # # Commenting this because I dont have this theme, feel free to uncomment 
    # plt.style.use('seaborn-darkgrid')
    plt.style.use("ggplot")
    # Crear dos columnas para colocar las gráficas
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        plt.figure(figsize=(8, 6))
        plt.plot(graph1["time"], graph1["Cplasma"], color="black")
        plt.xlabel("Time, days")
        plt.ylabel("Cplasma, uM")
        plt.title(
            f"Plasma concentration vs. time {st.session_state.names[st.session_state.count]}"
        )
        st.pyplot(plt)

    with fig_col2:
        plt.figure(figsize=(8, 6))
        plt.scatter(graph2["dose"], graph2["Css"], color="black")
        plt.plot(
            graph2[graph2["dose"] == 1]["dose"],
            graph2[graph2["dose"] == 1]["Css"],
            linestyle="--",
        )  # Línea punteada
        plt.plot(
            graph2["dose"], graph2["Css"], linestyle="--", color="black"
        )  # Línea sólida
        plt.xlabel("Dose")
        plt.ylabel("Css")
        plt.title(
            f"Css vs. daily dose of {st.session_state.names[st.session_state.count]}"
        )
        st.pyplot(plt)

    if on:
        st.markdown(
            '## <span style="color:#696969">TK Parameters</span>', unsafe_allow_html=True
        )
        parameters = ["AUC", "Peak", "Mean"]
        values = stats.values[0]
        for parameter, value in zip(parameters, values):
            st.markdown(f"<p style='font-size: 20px; color: #696969;'>- <strong>{parameter}:</strong> {value}</p>", unsafe_allow_html=True)


def next_quote():
    N = max(len(st.session_state.df_graph1), len(st.session_state.df_graph2))
    # Roll over if the next counter is going to be out of bounds
    if st.session_state.count + 1 == N:
        st.session_state.count = 0
    else:
        st.session_state.count += 1

def fingerprints_inputs(dataframe):

    X=np.array([AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,useFeatures=True) for mol in [Chem.MolFromSmiles(m) for m in list(dataframe.canonical_smiles)]])
    y=dataframe.pchembl_value.astype('float')
    return X,y

def fingerprints_inputs2(dataframe):
    X=np.array([AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,useFeatures=True) for mol in [Chem.MolFromSmiles(m) for m in list(dataframe.Smiles)]])
    y=dataframe.Activity.astype('int')
    return X,y

def previous_quote():
    N = max(len(st.session_state.df_graph1), len(st.session_state.df_graph2)) - 1
    # Roll back if the previous counter is going to be out of bounds
    if st.session_state.count == 0:
        st.session_state.count = N
    else:
        st.session_state.count -= 1

if selected == "Home":
    
    # st.markdown("<h1 style='text-align: center; fontSize: 30px; font-style: italic; color: grey;'>Mechanistic modeling allows for a better understanding of the biology underlying a toxicological outcome</h1>", unsafe_allow_html=True)
    # st.write('')
    # st.markdown("<h1 style='text-align: center; fontSize: 23px; color: grey; font-weight: normal;'>Obtain cholestasis predictions based on combinations of QSAR transporter models which representing simpler biological phenomena through integration of pharmacokinetic information.</h1>", unsafe_allow_html=True)
    # st.write('---')
    # st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>Summary information</h1>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">This web application enables predictions of cholestasis using a methodology that combines predictive QSAR models for various hepatic transporters. These models predict <em>in vitro</em> concentrations, which are then extrapolated to <em>in vivo</em> doses using QIVIVE models and compared with maximum therapeutic doses.</div>', unsafe_allow_html=True)   
    st.write('')
    st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>Funding sources</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify; fontSize: 18px; color: #696969;'>This application is the result of this published article: <a href='https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945'><u>https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945</u></a>.</div>", unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; font-size: 18px; color: #696969;">'
             'This study was developed during a contract period of my PhD, funded by eTRANSAFE ('
             '<a href="https://etransafe.eu/" target="_blank"><u>https://etransafe.eu/</u></a>)'
             ' and Risk-Hunt3r (<a href="https://www.risk-hunt3r.eu/" target="_blank"><u>https://www.risk-hunt3r.eu/</u></a>)'
             ' projects, both supported by European Union’s Horizon 2020 and the EFPIA.'
             '</div>', unsafe_allow_html=True)
    # st.write("StreamChol is an application useful for PK and Machine Learning analysis of drugs which could induce cholestasis. This project has been developed during a contract period of my PhD funded by eTRANSAFE project, Innovative Medicines Initiative 2 Joint Undertaking under Grant Agreement No. 777365, supported from European Union’s Horizon 2020 and the EFPIA. Authors declare that this work reflects only the author’s view, and that IMI-JU is not responsible for any use that may be made of the  information it contains. Also, this project received funding from the European Union’s Horizon 2020 Research and Innovation programme under Grant Agreement No. 964537 (RISK-HUNT3R), which is part of the ASPIS cluster.")
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <img src="https://raw.githubusercontent.com/phi-grib/flame/master/images/imi-logo.png" alt="IMI logo" width="200">
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<img src="https://github.com/phi-grib/flame/blob/master/images/eTRANSAFE-logo-git.png?raw=true" width="200" />', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<img src="https://raw.githubusercontent.com/phi-grib/namastox_web/master/images/risk-hunt3r-logo.png" width="150" />', unsafe_allow_html=True)

elif selected == "Tutorial":

    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;"><strong>StreamChol</strong> allows predicting cholestasis for one or more compounds. For a single compound, select the <strong>Smile and PK Information</strong> tab under the <strong>Data</strong> section. Here, you can input the Smile representation of compound or draw it using the Sketcher tool. Additionally, you will need to provide certain information such as the name, ChEMBL ID, Max Dose, Unbound fraction, and intrinsic clearance (CLint).</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">Alternatively, if you want to process multiple compounds, you can upload an <strong>Excel file</strong> with the following variables in the same order and format: name, ID, Smiles, Maximum Dose, Activity, FUB, CLint.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">Next, select the <strong>PK analysis</strong> tab to calculate steady-state plasma concentration. Here, you can choose the dose per day, the daily doses (number of doses), as well as the days of administration. You can also select whether you want additional TK information such as AUC, mean, and peak. After that, press the <strong>Run PK Analysis</strong> button, and you will get two types of graphs: one comparing Css vs time and another comparing Css at different doses, along with a dropdown for interpreting the graphs.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">Finally, in the <strong>Prediction</strong> tab, you can predict the cholestatic activity of the compound(s) in question. You can select hyperparameters such as a small correction factor on <em>in vivo</em> doses (K), the transporters you want to include in the analysis, and the logical rule you want to apply (OR, AND, Majority). Additionally, when predicting the activity of more than one compound.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">In the last tabs, <strong>Report</strong> and <strong>Contact</strong>, you can find information about the models used along with brief information about me and how to contact me if needed.</div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">Please, when you finish all your computations, clear cache.</div>', unsafe_allow_html=True)
    

elif selected == 'Report':

    # st.write('')
    st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>PK Analysis</h1>", unsafe_allow_html=True)
    st.write('')
    st.markdown("<div style='text-align: justify; fontSize: 18px; color: #696969;'>The first of the two PK analysis graphs, which compares steady-state plasma concentration vs. time, is obtained through the function calc_analytic_css from the HTTK library(<a href='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6134854/'><u>https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6134854/</u></a>). This function calculates the analytic steady-state plasma concentration resulting from infusion dosing using a multiple compartment PBTK model. </div>", unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">The second graph, CSS vs. Dose, is constructed using the aforementioned function in a loop at different doses (0.1, 0.5, 1.0, 1.5, 2.0) mg/kg bw/day. Finally, tkstats allows obtaining the area under the curve, the mean, and the peak values for the plasma concentration.</div>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>Metamodel</h1>", unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">The "metamodel" function allows for predictions of cholestasis by combining information from <em>in vivo</em> oral dose-response data of multiple hepatic transporters (BCRP, MRP2, MRP3, MRP4, OATP1B1, OATP1B3, BSEP, P-gp), which are extrapolated from <em>in vitro</em> concentrations (IC50). To obtain the <em>in vitro</em> concentrations, individual QSAR models were developed using Morgan fingerprints and a battery of machine learning models. Subsequently, the IC50 values are extrapolated to <em>in vivo</em> oral doses using the HTTK tool with the calc_mc_oral_equiv function. This function employs reverse dosimetry-based <em>in vitro</em>-<em>in vivo</em> extrapolation (IVIVE) for high throughput risk screening through a PBPK model. These oral doses are then compared with the maximum doses of each compound. In this function, we can assign a small correction factor (K) to the oral doses, select the transporters, and choose the logical rule for combining the information (OR, AND, Majority). The selected metrics (Sensitivity, Specificity, MCC, Accuracy, and ROC-AUC) are shown for both training series (a set of 426 compounds collected in our published work) and test series.</div>', unsafe_allow_html=True)


elif selected == 'Data':

    genre = st.radio(
    "Select input format",
    ["Smile and PK information", "Excel file"],captions=['Smile or sketcher with PK information', 'Introduce next variables: name, ID, Smiles, Doses max, Activity, FUB, CLint'],
    index=None,)
    
    
    st.markdown(
    """
    <style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)
    

    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] > p {
            font-size: 20px; /* Sketcher */
            color: #696969; /* Grey color */
            margin-top: -3px; /* Move the text upwards by 10px */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("You selected:", genre)
    
    if genre=='Smile and PK information':

        molecule = st.text_input("Molecule", 'CC(=O)Nc1ccc(O)cc1')

        st.markdown(
    """
    <style>
    div[class*="stTextInput"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)
        
        st.markdown(
    """
    <style>
    div[data-baseweb="input"] > div[data-baseweb="base-input"] > input {
        font-size: 20px;color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
        smile_code = st_ketcher(molecule)
        name = st.text_input("Drug name:",'Acetaminophen')
        chembl_id = st.text_input("ChEMBL ID:",'CHEMBL112')

        st.markdown(
    """
    <style>
    div[class*="stNumberInput"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)
          
        st.markdown(
    """
    <style>
    /* Estilos para el tamaño de la fuente del campo de entrada numérico */
    .st-emotion-cache-kskxxl input {
        font-size: 20px;color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
        max_dose = st.number_input('Max Dose in mg',min_value=1,max_value=100000)
        FUB = st.number_input('Unbound fraction (rate)',min_value=0.00001, max_value=1.0)
        clint=st.number_input('Intrinsic clearance in mL/min',min_value=0.00001)

        st.markdown(
    """
    <style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
        margin-top: -15px;
        margin-left: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True)
        
        st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stTickBarMin"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-left:-5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stTickBarMax"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-right:-5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stThumbValue"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-top: -7px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
        act= st.slider('Activity',min_value=0,max_value=1,key='Activity')

        # Crear un DataFrame con el resultado
        data = {'Name':[name],'ID': [chembl_id],'Smiles': [smile_code],'Doses max':[max_dose],'Activity':[act],'FUB':[FUB],'CLint':[clint]}
        df_result = pd.DataFrame(data)

        st.markdown(
    """
    <style>
    /* Estilos para mover el botón a la izquierda */
    .st-emotion-cache-7ym5gk.ef3psqc11 {
        margin-left: -10px; /* Adjust the value to move the button to the left */
        font-size:23px;
        color:#696969;
    }
    </style>
    """,
    unsafe_allow_html=True
)


        if st.button("Save molecule",key = 'mol'):
            st.session_state['df_result'] = df_result
            # st.session_state['molecule'] = molecule
            st.experimental_rerun()

    
    elif genre=='Excel file':
        file_uploaded = st.file_uploader("Select one file", type=["csv", "txt", "xlsx", "sdf"])

        if file_uploaded is not None:  # Verifica si se ha cargado un archivo
            contenido = file_uploaded.read()
            
            if file_uploaded.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df_uploaded = pd.read_excel(io.BytesIO(contenido), index_col=0, engine='openpyxl')
            else:
                st.error("Unsupported file format. Please upload an Excel file.")
        else:
            st.warning("Please upload a file before attempting to read it.")
        
        if st.button("Save dataframe",key = 'dataframe'):
            st.session_state['df_uploaded'] = df_uploaded
            st.experimental_rerun()

elif selected == 'PK analysis':

    st.markdown(
    """
    <style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
        margin-top: -15px;
        margin-left: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True)
        
    st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stTickBarMin"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-left:-5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stTickBarMax"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-right:-5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <style>
    /* Estilos para el tamaño de fuente y color del texto */
    [data-testid="stThumbValue"] {
        font-size: 20px; /* Increase font size for the text */
        color: #696969; /* Grey color */
        margin-top: -7px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    dai_dose = st.slider('Daily dose', min_value=1, step=1, key="pk1")
    dose_per_day = st.slider('Dose per day (number of doses per day)', min_value=1, step=1, key="pk2")
    day = st.slider('Days', min_value=1, step=1, key="pk3")

    st.markdown(
    """
    <style>
    /* Estilos para aumentar el tamaño de fuente del texto */
    .st-emotion-cache-1vbkxwb.e1nzilvr5 p {
        font-size: 20px; /* Increase font size to 24px */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    on = st.toggle('TK stats' , help='It provides information about AUC, Peak, Mean')
    
    # st.markdown("<div style='height: 0px'></div>", unsafe_allow_html=True)

    if "Run PK Analysis" not in st.session_state:
        st.session_state["Run PK Analysis"] = False

    if "PK interpretation" not in st.session_state:
        st.session_state["PK interpretation"] = False


    if st.button("Run PK Analysis", key="pk_button"):
        st.session_state["Run PK Analysis"] = not st.session_state["Run PK Analysis"]
        if 'pk_parameters' not in st.session_state:
            st.session_state['pk_parameters']=[dai_dose,dose_per_day,day]
        with st.spinner("Building graphs..."):

            time.sleep(5)
            if 'df_result' in st.session_state:
                
                df_result = st.session_state['df_result']
                
                X_test, y_test = fingerprints_inputs2(df_result)

                df_nuevo_def3=df_result[['FUB','CLint','Doses max','Activity']].copy()

                smi=df_result.Smiles.tolist()
                
                mwt=[rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(m)) for m in smi]
                
                df_nuevo_def3.insert(loc = 4,
                        column = 'MW',
                        value = mwt)
                
                logp=[rdkit.Chem.Crippen.MolLogP(Chem.MolFromSmiles(m)) for m in smi]
                df_nuevo_def3.insert(loc = 5,
                        column = 'LOGP',
                        value = logp)
                
                df_nuevo_def3['CAS_fake']=[f'0-0-0-{i}' for i in range(1)]
                df_nuevo_def3['DTXSID_fake']=[f'DTXSID-{i}' for i in range(1)]
                df_nuevo_def3['fake_name']=[f'A-{i}' for i in range(1)]


                # Convert DataFrame from pandas to a R DataFrame
                df_r = py2rpy_pandasdataframe(df_nuevo_def3)

                ro.globalenv['df_r'] = df_r
                
                with localconverter(default_converter) as cv:
                    dai=robjects.vectors.IntVector([dai_dose])
                    dose_per=robjects.vectors.IntVector([dose_per_day])
                    da=robjects.vectors.IntVector([day])
                    r.assign('dai',dai)
                    r.assign('dose_per',dose_per)
                    r.assign('da',da)

                script_r = r('''
                                library(httk)
                                library(stringr)
                                library(dplyr)
                                
                                my.new.data <- as.data.frame(df_r$fake_name,stringsAsFactors=FALSE)
                                my.new.data <- cbind(my.new.data,as.data.frame(df_r$CAS_fake,stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$DTXSID_fake),stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$MW)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$LOGP)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$FUB)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$CLint)))
                                
                                colnames(my.new.data) <- c("Name","CASRN","DTXSID","MW","LogP","Fup","CLint")
                                chem.physical_and_invitro.data <- add_chemtable(my.new.data,
                                current.table=
                                chem.physical_and_invitro.data,
                                data.list=list(Compound="Name",CAS="CASRN",DTXSID="DTXSID",MW="MW",logP="LogP",
                                Funbound.plasma="Fup",Clint="CLint"),overwrite=TRUE,species="Human",reference="httk|chembl|medscape|NIH")
                                sol_pbtk <- solve_model(chem.name = df_r$fake_name, # Chemical to simulate
                                            model = "pbtk", # TK model to use
                                            dosing = list(initial.dose = NULL, # For repeated dosing, if first dose is different from the rest, specify first dose here
                                                    doses.per.day = dose_per, # Number of doses per day
                                                    daily.dose = dai, # Total daily dose in mg/kg units
                                                    dosing.matrix = NULL), # Used to specify more complicated dosing protocols
                                            days = da) # Number of days to simulate
                                plot.data <- as.data.frame(sol_pbtk)

                                tkstats <- calc_tkstats(chem.name = df_r$fake_name, # Chemical to simulate
                                                stats = c("AUC", "peak", "mean"), # Which metrics to return (these are the only three choices)
                                                model = "pbtk", # Model to use
                                                tissue = "plasma", # Tissue for which to return internal dose metrics
                                                days = da, # Length of simulation
                                                daily.dose = dai, # Total daily dose in mg/kg/day
                                                doses.per.day = dose_per) # Number of doses per day
                                
                                tks <- data.frame(AUC = tkstats[1], peak = tkstats[2], mean = tkstats[3])
                             
                                doses <- c(0.1, 0.5, 1.0, 1.5, 2.0)
                                answer_css <- data.frame()
                                for (i in doses) {
                                    output <- calc_analytic_css(chem.name=df_r$fake_name,
                                                                tissue='liver',
                                                                species='rabbit',
                                                                parameterize.args = list(default.to.human=TRUE,
                                                                                        adjusted.Funbound.plasma=TRUE,
                                                                                        regression=TRUE,
                                                                                        minimum.Funbound.plasma=1e-4),
                                                                daily.dose=i)
                                    answer_css <- rbind(answer_css, data.frame(Css=output))
                                }
                                DF_graph2 <- data.frame(dose = doses, Css = answer_css$Css)
                              
                                ''')

                with localconverter(default_converter + pandas2ri.converter) as cv:
                    
                    dm_tks = robjects.conversion.rpy2py(robjects.r['tks'])

                with localconverter(default_converter + pandas2ri.converter) as cv:
                    
                    dm_listing = robjects.conversion.rpy2py(robjects.r['plot.data'])

                with localconverter(default_converter + pandas2ri.converter) as cv:
                    
                    df_graph3 = robjects.conversion.rpy2py(robjects.r['DF_graph2'])
                
                st.session_state["dm_tks"] = dm_tks
                st.session_state["dm_listing"] = dm_listing
                st.session_state["df_graph3"] = df_graph3

                # st.write(dm_listing)
                plt.style.use('seaborn-darkgrid')
                na=df_result['Name']

                col1,col2=st.columns(2)

                with col1:
                    plt.figure(figsize=(8, 6))
                    plt.plot(dm_listing['time'], dm_listing['Cplasma'],color='black')
                    plt.xlabel("Time, days")
                    plt.ylabel("Cplasma, uM")
                    plt.title(f"Plasma concentration vs. time for {na[0]}")
                    st.pyplot(plt)
                
                with col2:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(df_graph3['dose'], df_graph3['Css'],color='black')
                    plt.plot(df_graph3[df_graph3['dose'] == 1]['dose'], df_graph3[df_graph3['dose'] == 1]['Css'], linestyle='--')  # Línea punteada
                    plt.plot(df_graph3['dose'], df_graph3['Css'], linestyle='--', color='black')  # Línea sólida
                    plt.xlabel("Dose")
                    plt.ylabel("Css")
                    plt.title(f"Css vs. daily dose of {na[0]}")
                    st.pyplot(plt)
                
                
                if on:
                    st.markdown('## <span style="color:#696969">Showing some TK parameters</span> ', unsafe_allow_html=True)
                    
                    parameters = ['AUC', 'Peak','Mean']
                    
                    values = dm_tks.values[0]
                    
                    for parameter, value in zip(parameters, values):
                        st.markdown(f"<p style='font-size: 20px; color: #696969;'>- <strong>{parameter}:</strong> {value}</p>", unsafe_allow_html=True)

                    
            elif 'df_uploaded' in st.session_state:
                
                df_uploaded = st.session_state['df_uploaded']
                
                X_test, y_test = fingerprints_inputs2(df_uploaded)

                df_nuevo_def3=df_uploaded[['FUB','CLint','Doses max','Activity']].copy()

                smi=df_uploaded.Smiles.tolist()
                
                mwt=[rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(m)) for m in smi]
                
                df_nuevo_def3.insert(loc = 4,
                        column = 'MW',
                        value = mwt)
                
                logp=[rdkit.Chem.Crippen.MolLogP(Chem.MolFromSmiles(m)) for m in smi]
                df_nuevo_def3.insert(loc = 5,
                        column = 'LOGP',
                        value = logp)
                
                df_nuevo_def3['CAS_fake']=[f'0-0-0-{i}' for i in range(df_nuevo_def3.shape[0])]
                df_nuevo_def3['DTXSID_fake']=[f'DTXSID-{i}' for i in range(df_nuevo_def3.shape[0])]
                df_nuevo_def3['fake_name']=[f'A-{i}' for i in range(df_nuevo_def3.shape[0])]

                df_r = py2rpy_pandasdataframe(df_nuevo_def3)

                # Pasar el DataFrame de R a R
                ro.globalenv['df_r'] = df_r

                with localconverter(default_converter) as cv:
                    dai=robjects.vectors.IntVector([dai_dose])
                    dose_per=robjects.vectors.IntVector([dose_per_day])
                    da=robjects.vectors.IntVector([day])
                    r.assign('dai',dai)
                    r.assign('dose_per',dose_per)
                    r.assign('da',da)

                tkstats=[] 
                df_graph1=[] 
                df_graph2=[]

                for i in df_nuevo_def3.fake_name.values:

                    with localconverter(default_converter) as cv:
                        r.assign('na',i)

                    r('''
                        library(httk)
                        library(stringr)
                        library(dplyr)
                        
                        my.new.data <- as.data.frame(df_r$fake_name,stringsAsFactors=FALSE)
                        my.new.data <- cbind(my.new.data,as.data.frame(df_r$CAS_fake,stringsAsFactors=FALSE))
                        my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$DTXSID_fake),stringsAsFactors=FALSE))
                        my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$MW)))
                        my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$LOGP)))
                        my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$FUB)))
                        my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$CLint)))
                        
                        colnames(my.new.data) <- c("Name","CASRN","DTXSID","MW","LogP","Fup","CLint")
                        chem.physical_and_invitro.data <- add_chemtable(my.new.data,
                        current.table=
                        chem.physical_and_invitro.data,
                        data.list=list(Compound="Name",CAS="CASRN",DTXSID="DTXSID",MW="MW",logP="LogP",
                        Funbound.plasma="Fup",Clint="CLint"),overwrite=TRUE,species="Human",reference="httk|chembl|medscape|NIH")
                        sol_pbtk <- solve_model(chem.name = na, # Chemical to simulate
                                    model = "pbtk", # TK model to use
                                    dosing = list(initial.dose = NULL, # For repeated dosing, if first dose is different from the rest, specify first dose here
                                            doses.per.day = dose_per, # Number of doses per day
                                            daily.dose = dai, # Total daily dose in mg/kg units
                                            dosing.matrix = NULL), # Used to specify more complicated dosing protocols
                                    days = da) # Number of days to simulate
                        plot.data <- as.data.frame(sol_pbtk)

                        tkstats <- calc_tkstats(chem.name = na, # Chemical to simulate
                                        stats = c("AUC", "peak", "mean"), # Which metrics to return (these are the only three choices)
                                        model = "pbtk", # Model to use
                                        tissue = "plasma", # Tissue for which to return internal dose metrics
                                        days = da, # Length of simulation
                                        daily.dose = dai, # Total daily dose in mg/kg/day
                                        doses.per.day = dose_per) # Number of doses per day
                        
                        tks <- data.frame(AUC = tkstats[1], peak = tkstats[2], mean = tkstats[3])
                    
                        doses <- c(0.1, 0.5, 1.0, 1.5, 2.0)
                        answer_css <- data.frame()
                        for (i in doses) {
                            output <- calc_analytic_css(chem.name=na,
                                                        tissue='liver',
                                                        species='rabbit',
                                                        parameterize.args = list(default.to.human=TRUE,
                                                                                adjusted.Funbound.plasma=TRUE,
                                                                                regression=TRUE,
                                                                                minimum.Funbound.plasma=1e-4),
                                                        daily.dose=i)
                            answer_css <- rbind(answer_css, data.frame(Css=output))
                        }
                        DF_graph2 <- data.frame(dose = doses, Css = answer_css$Css)
                    
                        ''')
                    
                    with localconverter(default_converter + pandas2ri.converter) as cv:
                        
                        tkstats.append(robjects.conversion.rpy2py(robjects.r['tks']))

                    with localconverter(default_converter + pandas2ri.converter) as cv:
                        
                        df_graph1.append(robjects.conversion.rpy2py(robjects.r['plot.data']))

                    with localconverter(default_converter + pandas2ri.converter) as cv:
                        
                        df_graph2.append(robjects.conversion.rpy2py(robjects.r['DF_graph2']))

                st.session_state["tkstats"] = tkstats
                st.session_state["df_graph1"] = df_graph1
                st.session_state["df_graph2"] = df_graph2

                if "count" not in st.session_state:
                    st.session_state.count = 0

                # Almacenar df_graph1 en el estado de la sesión
                if "df_graph1" not in st.session_state:
                    st.session_state.df_graph1 = df_graph1

                if "df_graph2" not in st.session_state:
                    st.session_state.df_graph2 = df_graph2

                if "tkstats" not in st.session_state:
                    st.session_state.tkstats = tkstats

                if "names" not in st.session_state:
                    st.session_state.names = df_uploaded.name.tolist()

    # Create a set of items which must be there in the session state before the graphs can be displayed
    required_states = {"df_graph1", "df_graph2"}

    # Only display the graph and navigation buttons once the analysis is done at least once and the fields df_graph1, df_graph2 are populated in the session state
    if required_states - set(st.session_state.keys()) == set():    
        # Display the navigation buttons
        prev_btn, _, next_btn = st.columns([1, 12, 1])

        if prev_btn.button("⏮️"):
            previous_quote()

        if next_btn.button("⏭️"):
            next_quote()
    
        # Display the current graph
        display_current_graph(on)
    
    st.markdown(
    """
    <style>
    /* Estilos para el tamaño de la fuente y color del texto */
    .st-emotion-cache-16idsys p {
        font-size: 20px; /* Cambia el tamaño de la fuente aquí */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    page_bg_img = f"""
    <style>
    [data-testid="stMarkdownContainer"] {{
    font-size: 20px; /* Increase font size to 24px */
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    if st.session_state["Run PK Analysis"] and ('df_result' in st.session_state or 'df_uploaded' in st.session_state):
        with st.expander("PK interpretation"):
            html_code = f"""
            <style>
            .custom-text {{
                font-size: 20px;
                color: #696969;
                text-align: justify;
            }}
            </style>
            <p class='custom-text'>
            A high rapid <strong>absorption</strong> can result in sustained high plasma concentration, increasing the risk of toxicity. Slow <strong>distribution</strong> can cause tissue accumulation, elevating plasma concentrations and toxicity risk. Inadequate <strong>elimination</strong> may result in drug buildup, increasing toxicity potential due to prolonged presence in the body.
            <br><br>
            <strong>AUC</strong> indicates increased exposure to a drug over time, potentially leading to prolonged therapeutic effects or heightened risk of adverse reactions due to elevated drug concentrations in the bloodstream. Higher values of AUC signifies more extensive drug absorption, distribution, and/or decreased elimination, emphasizing the need for careful monitoring and adjustment of dosage regimens to optimize therapeutic outcomes and mitigate potential toxicity.
            <br><br>
            Regarding the introduced PK parameters in the model, a high <strong>Clint</strong> value suggests rapid drug elimination, leading to a quick decline in plasma concentrations, shorter therapeutic effects, and potentially lower toxicity. A high <strong>unbound fraction</strong> (fub) implies increased drug availability for receptor interaction, potentially enhancing therapeutic efficacy but also raising the risk of toxicity due to heightened exposure.
            </p>
            """

            st.markdown(html_code, unsafe_allow_html=True)
            

elif selected == 'Prediction':
    st.markdown(
    """
    <style>
    div[class*="stNumberInput"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)

    st.markdown("""
    <style>
    input[type=number] {
        font-size: 20px;
        color: #696969;
    }
    </style>
    """, unsafe_allow_html=True)
          
    k_met = st.number_input('K (factor for correcting in vivo doses)',value=4.696969696969697, key="pk5")
    # Definir las opciones disponibles y las opciones seleccionadas por defecto
    options = ['BCRP', 'MRP2', 'MRP3', 'MRP4','OATP1B1','OATP1B3','BSEP','PGP']
    default_options = ['BSEP']

    # Agregar la opción "Seleccionar todo" a las opciones disponibles
    options_with_select_all = ['Select All'] + options

    st.markdown(
    """
    <style>
    div[class*="stMultiSelect"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)
    
    # Crear el widget de selección múltiple con las opciones modificadas
    selected_options = st.multiselect(
        'Select transporter model',
        options=options_with_select_all,
        default=default_options
    )

    st.markdown(
    """
    <style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 23px;
        color: #696969; /* Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] > p {
            font-size: 20px; /* Ketcher */
            color: #696969; /* Grey color */
            margin-top: -3px; /* Move the text upwards by 10px */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    logical = st.radio("Choose an option", ('OR','Majority','AND'),captions=['Cholestasis = any of the in vivo doses of at least one transporter < max dose','Cholestasis = The majority of transporters with in vivo doses < max dose','Cholestasis = All transporters with in vivo doses < max dose'])
    

    st.markdown(
    """
    <style>
    /* Estilos para mover el botón a la izquierda */
    .st-emotion-cache-7ym5gk.ef3psqc11 {
        margin-left: 0px; /* Adjust the value to move the button to the left */
        font-size:23px;
        color:#696969;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    # if "Download data as CSV" not in st.session_state:
    #     st.session_state["Download data as CSV"] = False

    if st.button("Cholestasis prediction"):
        
        pk_parameters = st.session_state['pk_parameters']
        day_dose=pk_parameters[0]
        dose_per_day=pk_parameters[1]
        day=pk_parameters[2]

        with st.spinner("Predicting..."):

            time.sleep(5)

            training_series=pd.read_csv('final_predictions_app.csv')
            
            random.seed(46)
            
            model_bcrp=load('bcrp')["modelo"]

            model_mrp2=load('mrp2')["modelo"]

            model_mrp3=load('mrp3')["modelo"]

            model_mrp4=load('mrp4')["modelo"]

            model_oat1=load('oat1')["modelo"]

            model_oat2=load('oat2')["modelo"]

            model_bsep=load('bsep')["modelo"]

            model_pgp=load('pgp')["modelo"]

            if 'df_result' in st.session_state:
                df_result = st.session_state['df_result']
                
                X_test, y_test = fingerprints_inputs2(df_result)
            
                pred_bcrp = model_bcrp.predict(X_test)
                pred_mrp2 = model_mrp2.predict(X_test)
                pred_mrp3 = model_mrp3.predict(X_test)
                pred_mrp4 = model_mrp4.predict(X_test)
                pred_oat1 = model_oat1.predict(X_test)
                pred_oat2 = model_oat2.predict(X_test)
                pred_pgp = model_pgp.predict(X_test)
                pred_bsep = model_bsep.predict(X_test)
                
                
                df_in_vitro=pd.DataFrame(data={'BCRP':pred_bcrp,'MRP2':pred_mrp2,'MRP3':pred_mrp3,'MRP4':pred_mrp4,'OATP1B1':pred_oat1,'OATP1B3':pred_oat2,'BSEP':pred_bsep,
                                                'PGP':pred_pgp,'Activity':y_test.values},index=df_result.index)
                
                df_nuevo_def3=pd.concat([df_in_vitro.iloc[:,:-1],df_result[['FUB','CLint','Doses max','Activity']]],axis=1)
                
                smi=df_result.Smiles.tolist()
                
                mwt=[rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(m)) for m in smi]
                
                df_nuevo_def3.insert(loc = 12,
                        column = 'MW',
                        value = mwt)
                
                logp=[rdkit.Chem.Crippen.MolLogP(Chem.MolFromSmiles(m)) for m in smi]
                df_nuevo_def3.insert(loc = 13,
                        column = 'LOGP',
                        value = logp)
                
                df_nuevo_def3['CAS_fake']=[f'0-0-0-{i}' for i in range(1)]
                df_nuevo_def3['DTXSID_fake']=[f'DTXSID-{i}' for i in range(1)]
                df_nuevo_def3['fake_name']=[f'A-{i}' for i in range(1)]

                df_r = py2rpy_pandasdataframe(df_nuevo_def3)

                ro.globalenv['df_r'] = df_r

                with localconverter(default_converter) as cv:
                    dai=robjects.vectors.IntVector([day_dose])
                    dose_per=robjects.vectors.IntVector([dose_per_day])
                    da=robjects.vectors.IntVector([day])
                    r.assign('dai',dai)
                    r.assign('dose_per',dose_per)
                    r.assign('da',da)
                
                script_r = r('''
                                library(httk)
                                library(stringr)
                                library(dplyr)
                                
                                my.new.data <- as.data.frame(df_r$fake_name,stringsAsFactors=FALSE)
                                my.new.data <- cbind(my.new.data,as.data.frame(df_r$CAS_fake,stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$DTXSID_fake),stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$MW)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$LOGP)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$FUB)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$CLint)))
                                
                                colnames(my.new.data) <- c("Name","CASRN","DTXSID","MW","LogP","Fup","CLint")
                                chem.physical_and_invitro.data <- add_chemtable(my.new.data,
                                current.table=
                                chem.physical_and_invitro.data,
                                data.list=list(Compound="Name",CAS="CASRN",DTXSID="DTXSID",MW="MW",logP="LogP",
                                Funbound.plasma="Fup",Clint="CLint"),overwrite=TRUE,species="Human",reference="httk|chembl|medscape|NIH")
                                
                                df_r$BCRP <- as.numeric(df_r$BCRP)
                                set.seed(42)
                                answer_bcrp<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$BCRP)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_bcrp<-rbind(answer_bcrp,output)
                                colnames(answer_bcrp) <- c("0.9_oral_dose_bcrp")
                                answer_bcrp$DTXSID<-df_r$DTXSID_fake

                                df_r$MRP2 <- as.numeric(df_r$MRP2)
                                set.seed(42)
                                answer_mrp2<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$MRP2)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_mrp2<-rbind(answer_mrp2,output)
                                colnames(answer_mrp2) <- c("0.9_oral_dose_mrp2")
                                answer_mrp2$DTXSID<-df_r$DTXSID_fake

                                df_r$MRP3 <- as.numeric(df_r$MRP3)
                                set.seed(42)
                                answer_mrp3<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$MRP3)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_mrp3<-rbind(answer_mrp3,output)
                                colnames(answer_mrp3) <- c("0.9_oral_dose_mrp3")
                                answer_mrp3$DTXSID<-df_r$DTXSID_fake

                                df_r$MRP4 <- as.numeric(df_r$MRP4)
                                set.seed(42)
                                answer_mrp4<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$MRP4)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_mrp4<-rbind(answer_mrp4,output)
                                colnames(answer_mrp4) <- c("0.9_oral_dose_mrp4")
                                answer_mrp4$DTXSID<-df_r$DTXSID_fake

                                df_r$OATP1B1 <- as.numeric(df_r$OATP1B1)
                                set.seed(42)
                                answer_oat1<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$OATP1B1)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_oat1<-rbind(answer_oat1,output)
                                colnames(answer_oat1) <- c("0.9_oral_dose_oat1")
                                answer_oat1$DTXSID<-df_r$DTXSID_fake

                                df_r$OATP1B3 <- as.numeric(df_r$OATP1B3)
                                set.seed(42)
                                answer_oat2<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$OATP1B3)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_oat2<-rbind(answer_oat2,output)
                                colnames(answer_oat2) <- c("0.9_oral_dose_oat2")
                                answer_oat2$DTXSID<-df_r$DTXSID_fake

                                df_r$BSEP <- as.numeric(df_r$BSEP)
                                set.seed(42)
                                answer_bsep<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$BSEP)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_bsep<-rbind(answer_bsep,output)
                                colnames(answer_bsep) <- c("0.9_oral_dose_bsep")
                                answer_bsep$DTXSID<-df_r$DTXSID_fake

                                df_r$PGP <- as.numeric(df_r$PGP)
                                set.seed(42)
                                answer_pgp<-data.frame()
                                output = calc_mc_oral_equiv(conc = 10^(-df_r$PGP)*10^6,dtxsid=df_r$DTXSID_fake,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                answer_pgp<-rbind(answer_pgp,output)
                                colnames(answer_pgp) <- c("0.9_oral_dose_pgp")
                                answer_pgp$DTXSID<-df_r$DTXSID_fake

                                dataframe_together_pred<-Reduce(function(x, y) left_join(x, y, by = "DTXSID"), list(answer_bcrp,answer_mrp2,answer_mrp3,answer_mrp4,answer_bsep,answer_oat1,answer_oat2,answer_pgp))
                                
                            ''')
                
                with localconverter(default_converter + pandas2ri.converter) as cv:
                    dm_qivive = robjects.conversion.rpy2py(robjects.r['dataframe_together_pred'])
                
                dm_qivive=dm_qivive.set_index(df_nuevo_def3.index).rename(columns={'DTXSID':'DTXSID_fake','0.9_oral_dose_bcrp':'BCRP','0.9_oral_dose_mrp2':'MRP2','0.9_oral_dose_mrp3':'MRP3',
                                                         '0.9_oral_dose_mrp4':'MRP4','0.9_oral_dose_bsep':'BSEP','0.9_oral_dose_oat1':'OATP1B1',
                                                         '0.9_oral_dose_oat2':'OATP1B3','0.9_oral_dose_pgp':'PGP'})

                comb = pd.concat([df_result[['Name','ID','Smiles','Doses max']],dm_qivive.drop(columns='DTXSID_fake'),df_result[['Activity']]],axis=1)
                # st.write(comb)


                class LogicalOrEstimatorpk(BaseEstimator, TransformerMixin):
                    def __init__(self, k=1, transporters=['BCRP', 'MRP2', 'MRP3', 'MRP4', 'BSEP', 'OATP1B1', 'OATP1B3', 'PGP','Select All'], logical_rule=['OR','Majority','AND']):
                        self.k = k
                        self.transporters=transporters
                        self.logical_rule=logical_rule
                    
                    def fit(self, X, y):
                        return self
                    
                    def predict(self, X):
                        if len(self.transporters) == 1 and self.transporters[0] != 'Select All':
                            pred = np.where(X['Doses max'].values > self.k * X[self.transporters[0]].values, 1, 0)
                        elif len(self.transporters) == 2:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1) > len(self.transporters) / 2, 1, 0)
                        elif len(self.transporters) > 2 and len(self.transporters) < 8:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1) > len(self.transporters) / 2, 1, 0)
                        elif 'Select All' in self.transporters:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1) > 8 / 2, 1, 0)
                        else:
                            raise ValueError("Invalid transporter combination and/or logical rules.")
                        return pred
                    def get_params(self, deep=True):
                        return {"k": self.k,'transporters':self.transporters,'logical_rule':self.logical_rule}
                    
                    def set_params(self, **parameters):
                        for parameter, value in parameters.items():
                            setattr(self, parameter, value)
                        return self
                
                train_preds = LogicalOrEstimatorpk(k=k_met,transporters=selected_options[0],logical_rule=logical).fit(training_series,training_series.Activity).predict(training_series)

                preds = LogicalOrEstimatorpk(k=k_met,transporters=selected_options[0],logical_rule=logical).fit(comb,comb.Activity).predict(comb)

                sens=sensitivity(training_series.Activity,train_preds)
                spe=especificity(training_series.Activity,train_preds)
                mcc=matthews_corrcoef(training_series.Activity,train_preds)
                acc=accuracy_score(training_series.Activity,train_preds)
                roc=roc_auc_score(training_series.Activity,train_preds)

                metrics_df = pd.DataFrame({
                'Metric': ['Accuracy','Sensitivity', 'Specificity', 'MCC', 'ROC-AUC'],
                'Score': [acc, sens, spe, mcc, roc]
            })
                st.markdown(f'<div style="text-align: justify; fontSize: 23px; color: #696969;"><strong>Training series metrics</strong> </div>', unsafe_allow_html=True)
                # Create a bar plot using Matplotlib
                plt.figure(figsize=(8, 6))
                plt.bar(metrics_df['Metric'], metrics_df['Score'], color=['blue', 'green', 'red', 'purple','orange'])
                plt.xlabel('')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Display the plot using Streamlit
                st.pyplot(plt)
                

                mol = Chem.MolFromSmiles(df_result.Smiles[0])
                st.markdown(f'<div style="text-align: justify; fontSize: 23px; color: #696969;"><strong>{df_result.Name.values[0]}</strong> </div>', unsafe_allow_html=True)
                # st.markdown(f'<p style="font-size: 23px; color: #696969;">{df_result.Name.values[0]}</p>', unsafe_allow_html=True)                
                img = Draw.MolToImage(mol)
                bio = BytesIO()
                st.image(img)
                
                st.markdown('## <span style="color:#696969">Extrapolated in vivo doses in mg/kg BW/ day</span>', unsafe_allow_html=True)
                parameters = ['BCRP','MRP2', 'MRP3','MRP4','BSEP','OATP1B1','OATP1B3','P-gp']
                values = comb.iloc[:,4:-1].values[0]
                for parameter, value in zip(parameters, values):
                    st.markdown(f"<p style='font-size: 20px; color: #696969;'>- <strong>{parameter}:</strong> {value}</p>", unsafe_allow_html=True)
                
                st.markdown('## <span style="color:#696969">Some Pk parameters</span>', unsafe_allow_html=True)
                parameters = ['Max dose (mg)', 'Fub', 'CLint (ml/min)']
                values = df_result[['Doses max','FUB','CLint']].values[0]
                for parameter, value in zip(parameters, values):
                    st.markdown(f"<p style='font-size: 20px; color: #696969;'>- <strong>{parameter}:</strong> {value}</p>", unsafe_allow_html=True)
                
                st.markdown('## <span style="color:#696969">Prediction</span>', unsafe_allow_html=True)
                # Verificar si preds coincide con comb.Activity
                if (preds == comb.Activity).all() and preds.all() == 1:
                    # Assign green color
                    style_icon = "font-weight: bold; font-size: larger; color: green; display: inline-block; margin-right: 10px;"
                    style_text = "font-weight: bold; font-size: 20px; color: green; display: inline-block;"
                    st.markdown(f'<p style="{style_icon}">✅</p><p style="{style_text}">Cholestatic</p>', unsafe_allow_html=True)
                elif (preds == comb.Activity).all() and preds.all() == 0:
                    # Assign green color
                    style_icon = "font-weight: bold; font-size: larger; color: green; display: inline-block; margin-right: 10px;"
                    style_text = "font-weight: bold; font-size: 20px; color: green; display: inline-block;"
                    st.markdown(f'<p style="{style_icon}">✅</p><p style="{style_text}">Non Cholestatic</p>', unsafe_allow_html=True)
                elif (preds != comb.Activity).all() and preds.all() == 1:

                    style_icon = "font-weight: bold; font-size: larger; color: red; display: inline-block; margin-right: 10px;"
                    style_text = "font-weight: bold; font-size: 20px; color: red; display: inline-block;"
                    st.markdown(f'<p style="{style_icon}">❌</p><p style="{style_text}">Cholestatic</p>', unsafe_allow_html=True)
                else:

                    style_icon = "font-weight: bold; font-size: larger; color: red; display: inline-block; margin-right: 10px;"
                    style_text = "font-weight: bold; font-size: 20px; color: red; display: inline-block;"
                    st.markdown(f'<p style="{style_icon}">❌</p><p style="{style_text}">Non Cholestatic</p>', unsafe_allow_html=True)

            elif 'df_uploaded' in st.session_state:

                df_uploaded = st.session_state['df_uploaded']
                n = df_uploaded.shape[0]
                X_test, y_test = fingerprints_inputs2(df_uploaded)
            
                pred_bcrp = model_bcrp.predict(X_test)
                pred_mrp2 = model_mrp2.predict(X_test)
                pred_mrp3 = model_mrp3.predict(X_test)
                pred_mrp4 = model_mrp4.predict(X_test)
                pred_oat1 = model_oat1.predict(X_test)
                pred_oat2 = model_oat2.predict(X_test)
                pred_pgp = model_pgp.predict(X_test)
                pred_bsep = model_bsep.predict(X_test)
                
                
                df_in_vitro=pd.DataFrame(data={'BCRP':pred_bcrp,'MRP2':pred_mrp2,'MRP3':pred_mrp3,'MRP4':pred_mrp4,'OATP1B1':pred_oat1,'OATP1B3':pred_oat2,'BSEP':pred_bsep,
                                                'PGP':pred_pgp,'Activity':y_test.values},index=df_uploaded.index)
                
                df_nuevo_def3=pd.concat([df_in_vitro.iloc[:,:-1],df_uploaded[['FUB','CLint','Doses max','Activity']]],axis=1)
                
                smi=df_uploaded.Smiles.tolist()
                
                mwt=[rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(m)) for m in smi]
                
                df_nuevo_def3.insert(loc = 12,
                        column = 'MW',
                        value = mwt)
                
                logp=[rdkit.Chem.Crippen.MolLogP(Chem.MolFromSmiles(m)) for m in smi]
                df_nuevo_def3.insert(loc = 13,
                        column = 'LOGP',
                        value = logp)
                
                df_nuevo_def3['CAS_fake']=[f'0-0-0-{i}' for i in range(n)]
                df_nuevo_def3['DTXSID_fake']=[f'DTXSID-{i}' for i in range(n)]
                df_nuevo_def3['fake_name']=[f'A-{i}' for i in range(n)]

                # Convertir el DataFrame de pandas a DataFrame de R
                df_r = py2rpy_pandasdataframe(df_nuevo_def3)

                # Pasar el DataFrame de R a R
                ro.globalenv['df_r'] = df_r

                with localconverter(default_converter) as cv:
                    dai=robjects.vectors.IntVector([day_dose])
                    dose_per=robjects.vectors.IntVector([dose_per_day])
                    da=robjects.vectors.IntVector([day])
                    r.assign('dai',dai)
                    r.assign('dose_per',dose_per)
                    r.assign('da',da)

                script_r = r('''
                                library(httk)
                                library(stringr)
                                library(dplyr)
                                
                                my.new.data <- as.data.frame(df_r$fake_name,stringsAsFactors=FALSE)
                                my.new.data <- cbind(my.new.data,as.data.frame(df_r$CAS_fake,stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$DTXSID_fake),stringsAsFactors=FALSE))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$MW)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$LOGP)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$FUB)))
                                my.new.data <- cbind(my.new.data,as.data.frame(c(df_r$CLint)))
                                
                                colnames(my.new.data) <- c("Name","CASRN","DTXSID","MW","LogP","Fup","CLint")
                                chem.physical_and_invitro.data <- add_chemtable(my.new.data,
                                current.table=
                                chem.physical_and_invitro.data,
                                data.list=list(Compound="Name",CAS="CASRN",DTXSID="DTXSID",MW="MW",logP="LogP",
                                Funbound.plasma="Fup",Clint="CLint"),overwrite=TRUE,species="Human",reference="httk|chembl|medscape|NIH")
                                
                                set.seed(42)
                                idx_list <- list(10^(-df_r$BCRP)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_bcrp<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_bcrp<-rbind(answer_bcrp,output)
                                }
                                colnames(answer_bcrp) <- c("0.9_oral_dose_bcrp")
                                answer_bcrp$DTXSID<-df_r$DTXSID_fake

                                set.seed(42)
                                idx_list <- list(10^(-df_r$MRP2)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_mrp2<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_mrp2<-rbind(answer_mrp2,output)
                                }
                                colnames(answer_mrp2) <- c("0.9_oral_dose_mrp2")
                                answer_mrp2$DTXSID<-df_r$DTXSID_fake
                                
                                set.seed(42)
                                idx_list <- list(10^(-df_r$MRP3)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_mrp3<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_mrp3<-rbind(answer_mrp3,output)
                                }
                                colnames(answer_mrp3) <- c("0.9_oral_dose_mrp3")
                                answer_mrp3$DTXSID<-df_r$DTXSID_fake
                                
                                set.seed(42)
                                idx_list <- list(10^(-df_r$MRP4)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_mrp4<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_mrp4<-rbind(answer_mrp4,output)
                                }
                                colnames(answer_mrp4) <- c("0.9_oral_dose_mrp4")
                                answer_mrp4$DTXSID<-df_r$DTXSID_fake

                                set.seed(42)
                                idx_list <- list(10^(-df_r$OATP1B1)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_oat1<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_oat1<-rbind(answer_oat1,output)
                                }
                                colnames(answer_oat1) <- c("0.9_oral_dose_oat1")
                                answer_oat1$DTXSID<-df_r$DTXSID_fake

                                set.seed(42)
                                idx_list <- list(10^(-df_r$OATP1B3)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_oat2<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_oat2<-rbind(answer_oat2,output)
                                }
                                colnames(answer_oat2) <- c("0.9_oral_dose_oat2")
                                answer_oat2$DTXSID<-df_r$DTXSID_fake


                                set.seed(42)
                                idx_list <- list(10^(-df_r$BSEP)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_bsep<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_bsep<-rbind(answer_bsep,output)
                                }
                                colnames(answer_bsep) <- c("0.9_oral_dose_bsep")
                                answer_bsep$DTXSID<-df_r$DTXSID_fake

                                set.seed(42)
                                idx_list <- list(10^(-df_r$PGP)*10^6, df_r$DTXSID_fake)
                                idx_vect <- c(1:length(idx_list[[1]]))
                                answer_pgp<-data.frame()
                                equiv_dose<-for(i in idx_vect){
                                    x <- idx_list[[1]][i]
                                    j <- idx_list[[2]][i]
                                    output = calc_mc_oral_equiv(conc = x,dtxsid=j,which.quantile = c(0.9),model="pbtk", daily.dose =dai, calc.analytic.css.arg.list =list(doses.per.day=dose_per, days=da))
                                    answer_pgp<-rbind(answer_pgp,output)
                                }
                                colnames(answer_pgp) <- c("0.9_oral_dose_pgp")
                                answer_pgp$DTXSID<-df_r$DTXSID_fake

                                dataframe_together_pred<-Reduce(function(x, y) left_join(x, y, by = "DTXSID"), list(answer_bcrp,answer_mrp2,answer_mrp3,answer_mrp4,answer_bsep,answer_oat1,answer_oat2,answer_pgp))

                                ''')
                
                with localconverter(default_converter + pandas2ri.converter) as cv:
                    dm_qivive = robjects.conversion.rpy2py(robjects.r['dataframe_together_pred'])
                
                dm_qivive=dm_qivive.set_index(df_nuevo_def3.index).rename(columns={'DTXSID':'DTXSID_fake','0.9_oral_dose_bcrp':'BCRP','0.9_oral_dose_mrp2':'MRP2','0.9_oral_dose_mrp3':'MRP3',
                                                         '0.9_oral_dose_mrp4':'MRP4','0.9_oral_dose_bsep':'BSEP','0.9_oral_dose_oat1':'OATP1B1',
                                                         '0.9_oral_dose_oat2':'OATP1B3','0.9_oral_dose_pgp':'PGP'})

                comb = pd.concat([df_uploaded[['name','ID','Smiles','Doses max']],dm_qivive.drop(columns='DTXSID_fake'),df_uploaded[['Activity']]],axis=1)


                class LogicalOrEstimatorpk(BaseEstimator, TransformerMixin):
                    def __init__(self, k=1, transporters=['BCRP', 'MRP2', 'MRP3', 'MRP4', 'BSEP', 'OATP1B1', 'OATP1B3', 'PGP','Select All'], logical_rule=['OR','Majority','AND']):
                        self.k = k
                        self.transporters=transporters
                        self.logical_rule=logical_rule
                    
                    def fit(self, X, y):
                        return self
                    
                    def predict(self, X):
                        # pred=None
                        if len(self.transporters) == 1 and self.transporters[0] != 'Select All':
                            pred = np.where(X['Doses max'].values > self.k * X[self.transporters[0]].values, 1, 0)
                        elif len(self.transporters) == 2:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1) > len(self.transporters) / 2, 1, 0)
                        elif len(self.transporters) > 2 and len(self.transporters) < 8:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X[self.transporters].values, axis=1) > len(self.transporters) / 2, 1, 0)
                        elif 'Select All' in self.transporters:
                            if self.logical_rule == 'OR':
                                pred = np.where(np.any(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1), 1, 0)
                            elif self.logical_rule == 'AND':
                                pred = np.where(np.all(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1), 1, 0)
                            elif self.logical_rule == 'Majority':
                                pred = np.where(np.sum(X['Doses max'].values[:, np.newaxis] > self.k * X.iloc[:, 4:-1].values, axis=1) > 8 / 2, 1, 0)
                        else:
                            raise ValueError("Invalid transporter combination and/or logical rules.")
                        # Manejar el caso en el que pred sigue siendo None
                        # if pred is None:
                        #     raise ValueError("No se pudo calcular 'pred' para los parámetros proporcionados.")
                        
                        return pred
                    def get_params(self, deep=True):
                        return {"k": self.k,'transporters':self.transporters,'logical_rule':self.logical_rule}
                    
                    def set_params(self, **parameters):
                        for parameter, value in parameters.items():
                            setattr(self, parameter, value)
                        return self
                
                train_preds = LogicalOrEstimatorpk(k=k_met,transporters=selected_options[0],logical_rule=logical).fit(training_series,training_series.Activity).predict(training_series)
                
                preds = LogicalOrEstimatorpk(k=k_met,transporters=selected_options[0],logical_rule=logical).fit(comb,comb.Activity).predict(comb)

                
                # report(comb.Activity, preds)

                metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'MCC','ROC-AUC']
                labels = ['Train', 'Test']
                x = np.arange(len(metrics))

                st.markdown(f'<div style="text-align: justify; fontSize: 23px; color: #696969;"><strong>Training and test series metrics</strong> </div>', unsafe_allow_html=True)
                sens_train=sensitivity(training_series.Activity,train_preds)
                spe_train=especificity(training_series.Activity,train_preds)
                mcc_train=matthews_corrcoef(training_series.Activity,train_preds)
                acc_train=accuracy_score(training_series.Activity,train_preds)
                roc_train=roc_auc_score(training_series.Activity,train_preds)

                sens_test=sensitivity(comb.Activity,preds)
                spe_test=especificity(comb.Activity,preds)
                mcc_test=matthews_corrcoef(comb.Activity,preds)
                acc_test=accuracy_score(comb.Activity,preds)
                roc_test=roc_auc_score(comb.Activity,preds)

                train_values = [acc_train, sens_train, spe_train, mcc_train, roc_train]
                test_values = [acc_test, sens_test, spe_test, mcc_test, roc_test]


                bar_width = 0.35


                fig, ax = plt.subplots(figsize=(10, 6))
                train_bars = ax.bar(x - bar_width/2, train_values, bar_width, label='Train')
                test_bars = ax.bar(x + bar_width/2, test_values, bar_width, label='Test')



                ax.set_ylabel('Score')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.legend()


                        
                autolabel(train_bars)
                autolabel(test_bars)


                st.pyplot(fig)
                
                comb['Predictions']=preds
                
                csv = comb.to_csv(index=False)

                b64 = base64.b64encode(csv.encode()).decode()  # some strings
                
                linko= f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
                
                st.markdown(linko, unsafe_allow_html=True)

                comb["img"] = comb["Smiles"].apply(smi_to_png)

                st.dataframe(comb, column_config={"img": st.column_config.ImageColumn()})

elif selected== 'Contact':

    col1, col2 = st.columns([1.25,1])

    with col1:
        st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>About me</h1>", unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">My name is Pablo Rodríguez Belenguer, and I work as a postdoctoral researcher in the PharmacoInformatics (PhI) group at Pompeu Fabra University in Barcelona, Spain, under the supervision of Team Leader Manuel Pastor Maeso. My academic background is in pharmacy, and I later received training in data science and artificial intelligence. My doctoral thesis focused on combining machine learning models to address complexities in the field of computational toxicology, encompassing biological, chemical, and methodological aspects. Throughout my tenure in the PhI group, I have come to understand that the key lies in comprehensively understanding the data and the biological foundations of the toxicological endpoints we aim to evaluate.</div>', unsafe_allow_html=True)
        
    with col2:
        left_container = """
        <div style="float: left; margin-right: 1rem; margin-top: 90px;">
            <img src="https://raw.githubusercontent.com/PARODBE/streamlit_figures/main/me.png" alt="Juntas" width="300" heigh="200">
        </div>
        """
        st.markdown(left_container, unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align: justify; fontSize: 18px; color: #696969;">If you share my passion for applying AI to toxicology, chemistry, or healthcare, please do not hesitate to reach out to me. I would be delighted to discuss further with you!</div>', unsafe_allow_html=True)
    st.write('---')
    st.markdown("<h1 style='text-align: left; fontSize: 30px; font-style: bold; color: #696969;'>Contact information</h1>", unsafe_allow_html=True)
    st.write('')
    st.markdown('[<img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" alt="Gmail Icon" width="30">](mailto:parodbe@gmail.com)', unsafe_allow_html=True)
    st.write('')
    st.markdown('[<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn Icon" width="30">](https://www.linkedin.com/in/pablorodriguezbelenguer)', unsafe_allow_html=True)
    st.write('')
    st.markdown('[<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub Icon" width="30">](https://github.com/parodbe)', unsafe_allow_html=True)
