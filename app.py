from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', ['AUC-ROC', 'AUC-PR', 'VUS-ROC', 'VUS-PR'])
    
tab_acc, tab_time, tab_stats = st.tabs(["Accuracy", "Execution Time", "Datasets"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    st.markdown(metric_name)
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    
