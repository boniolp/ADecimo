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
    if len(df.columns) > 0:
        fig = plt.figure(figsize=(10, min(30,int(0.40*len(df.columns)))))
        order = list(df_toplot.median().sort_values().index)[::-1]
        sns.boxplot(data=df_toplot,order=order,showfliers = False, orient='h') 
        st.pyplot(fig)
    
def generate_dataframe(df,datasets,methods_family,length,type_exp='_score'):
    if type_exp == '_score':
        selected_methods = old_method
    elif type_exp == '_inf':
        selected_methods = []
        
    if 'Transformer' in methods_family:
        selected_methods += [method.format(length).replace('_score',type_exp) for method in methods_sit]
    elif 'Convolutional' in methods_family:
        selected_methods += [method.format(length).replace('_score',type_exp) for method in methods_conv]
    elif 'Rocket' in methods_family:
        selected_methods += [method.format(length).replace('_score',type_exp) for method in methods_ts]
    elif 'Features' in methods_family:
        selected_methods += [method.format(length).replace('_score',type_exp) for method in methods_classical]
    
    return df.loc[df['dataset'].isin(datasets)][selected_methods]
        
    
    
with st.sidebar:
    st.markdown('# ADecimo') 
    st.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.selectbox('Pick an accuracy measure', list_measures)
    
    container_dataset = st.container()  
    all_dataset = st.checkbox("Select all",key='all_dataset')
    if all_dataset: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values)), list(set(df['dataset'].values)))
    else: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values))) 
    
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
    df_toplot = generate_dataframe(df,datasets,methods_family,length,type_exp='_score')
    st.dataframe(df_toplot)
    plot_box_plot(df_toplot)
    
with tab_time:
    st.markdown('# Execution Time Evaluation')
    st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark. Measure used: Prediction time in seconds')
    df_toplot = generate_dataframe(df,datasets,methods_family,length,type_exp='_inf')
    st.dataframe(df_toplot)
    plot_box_plot(df_toplot)
    
with tab_stats:
    st.markdown('# Dataset Statistics')
    st.dataframe(df[dataset_stats])
    
