"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st


def show_sidebar(df, list_measures, method_group, list_length):
    """
    Displays the sidebar for user interaction.

    Args:
        df (DataFrame): DataFrame containing data.
        list_measures (list): List of accuracy measures.
        method_group (dict): Dictionary containing method groups.
        list_length (list): List of lengths.

    Returns:
        tuple: Tuple containing selected metric name, datasets, method families, and length.
    """
    st.sidebar.markdown('# ADecimo') 
    st.sidebar.markdown('### Model selection for time series anomaly detection') 
    metric_name = st.sidebar.selectbox('Pick an accuracy measure', list_measures)
    
    container_dataset = st.sidebar.container()  
    all_dataset = st.sidebar.checkbox("Select all", key='all_dataset')
    if all_dataset:
        datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values)), list(set(df['dataset'].values)))
    else:
        datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values))) 
    
    container_method = st.sidebar.container()
    all_method = st.sidebar.checkbox("Select all", key='all_method')
    if all_method:
        methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()), list(method_group.keys()), key='selector_methods_all')
    else:
        methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()), key='selector_methods')
    
    container_length = st.sidebar.container()
    all_length = st.sidebar.checkbox("Select all", key='all_length')
    if all_length:
        length = container_length.multiselect('Select a window length', list_length, list_length, key='selector_length_all')
    else:
        length = container_length.multiselect('Select a window length', list_length, key='selector_length')
    
    return metric_name, datasets, methods_family, length



""" # Streamlit sidebar for user interaction
with st.sidebar:
	st.markdown('# ADecimo') 
	st.markdown('### Model selection for time series anomaly detection') 
	metric_name = st.selectbox('Pick an accuracy measure', list_measures)
	
	container_dataset = st.container()  
	all_dataset = st.checkbox("Select all", key='all_dataset')
	if all_dataset: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values)), list(set(df['dataset'].values)))
	else: datasets = container_dataset.multiselect('Select Datasets', list(set(df['dataset'].values))) 
	
	container_method = st.container()
	all_method = st.checkbox("Select all", key='all_method')
	if all_method: methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()), list(method_group.keys()), key='selector_methods_all')
	else: methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()), key='selector_methods')
	
	container_length = st.container()
	all_length = st.checkbox("Select all", key='all_length')
	if all_length: length = container_length.multiselect('Select a window length', list_length, list_length, key='selector_length_all')
	else: length = container_length.multiselect('Select a window length', list_length, key='selector_length')
 """