from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    
tab1, tab2 = st.tabs(["Accuracy", "Execution Time"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    

