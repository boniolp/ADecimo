from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

from constant import *

with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', list_measures)
 
df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))

tab_acc, tab_time, tab_stats = st.tabs(["Accuracy", "Execution Time", "Datasets"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    st.markdown('measure used: {}'.format(metric_name))
    st.dataframe(df[[method + '_score' for method in methods]])
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    st.dataframe(df[[method + '_inf' for method in methods]])
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    st.dataframe(df[dataset_stats])
    
