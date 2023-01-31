from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

from constant import *

with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', ['VUS-PR','AUC-PR','VUS-ROC','AUC-ROC'])
    
tab_acc, tab_time, tab_stats = st.tabs(["Accuracy", "Execution Time", "Datasets"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    st.markdown('measure used: {}'.format(metric_name))
    df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
    st.dataframe(df)
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    
