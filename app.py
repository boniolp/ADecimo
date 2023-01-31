from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

from constant import *

#plt.style.use('dark_background')

df = pd.read_csv('data/merged_scores_{}.csv'.format('VUS_PR'))
df = df.set_index('filename')

def plot_box_plot(df):
    fig = plt.figure(figsize=(10, 30))
    order = list(df_toplot.median().sort_values().index)[::-1]
    sns.boxplot(data=df_toplot,order=order,showfliers = False, orient='h') 
    st.pyplot(fig)
    

with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', list_measures)
    
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all",key='all_dataset')
    if all_dataset: container_dataset = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values)), list(set(df['dataset'].values)))
    else: container_dataset = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values))) 
    
    container_method = st.container()
    all_method = st.checkbox("Select all",key='all_method')
    if all_method: methods_family = container_method.multiselect('Select a group of methods', ['Transformer','Convolutional','Rocket','Features'], ['Transformer','Convolutional','Rocket','Features'])
    else: methods_family = container_method.multiselect('Select a group of methods', ['Transformer','Convolutional','Rocket','Features'])
    
    container_length = st.container()
    all_length = st.checkbox("Select all",key='all_length')
    if all_length: length = container_length.multiselect('Select a window length', list_length, list_length)
    else: length = container_length.multiselect('Select a window length', list_length)

df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
df = df.set_index('filename')

tab_acc, tab_time, tab_stats = st.tabs(["Accuracy", "Execution Time", "Datasets"])  
with tab_acc:
    st.markdown('# Accuracy Evaluation')
    st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark. Measure used: {}'.format(metric_name))
    df_toplot = df.loc[df['dataset'].isin(datasets)][[method + '_score' for method in methods] + old_method]
    st.dataframe(df_toplot)
    plot_box_plot(df_toplot)
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    df_toplot = df.loc[df['dataset'].isin(datasets)][[method + '_inf' for method in methods]]
    st.dataframe(df_toplot)
    plot_box_plot(df_toplot)
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    st.dataframe(df[dataset_stats])
    
