from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

from constant import *

with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', list_measures)
 
df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
df = df.set_index('filename')

tab_acc, tab_time, tab_stats = st.tabs(["Accuracy", "Execution Time", "Datasets"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    st.markdown('measure used: {}'.format(metric_name))
    df_toplot = df[[method + '_score' for method in methods] + old_method]
    st.dataframe(df_toplot)
    
    fig = plt.figure(figsize=(10, 4))
    order = list(df_toplot.median().sort_values().index)[::-1]
    sns.boxplot(data=df_toplot,order=order,showfliers = False)
    st.pyplot(fig)
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    st.dataframe(df[[method + '_inf' for method in methods] + old_method])
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    st.dataframe(df[dataset_stats])
    
