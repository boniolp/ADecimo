"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""

import streamlit as st
import pandas as pd

from utils.constant import list_length, all_datasets, method_group
from utils.helper import generate_dataframe, plot_box_plot

# Load score data
df = pd.read_csv('data/merged_scores_VUS_PR.csv')  # Assuming scores are stored with a fixed identifier
df = df.set_index('filename')

# Load inference time data
df_time = pd.read_csv('data/inference_time.csv')
df_time = df_time.rename(columns={'Unnamed: 0': 'filename'})
df_time = df_time.set_index('filename')

# Load training time data
df_time_train = pd.read_csv('data/training_times.csv', index_col='window_size')

# Create columns for selections
col_dataset_over, col_method_over, col_length_over = st.columns(3)  # Use unpacking if you prefer that style

# Dataset selection
with col_dataset_over:
	datasets = st.multiselect('Pick datasets', 
							  all_datasets, 
							  help="Select one or more datasets for analysis.")

# Method selection
with col_method_over:
	methods_family = st.multiselect('Pick methods', 
									list(method_group.keys()), 
									help="Select one or more method groups for comparison.")

# Window length selection
with col_length_over:
	length = st.multiselect('Pick window lengths', 
							list_length, 
							help="Select the time window lengths applicable to the selected methods.")

# Define tabs for different time evaluations
tab_training, tab_prediction, tab_inference = st.tabs(["Training Time", "Selection Time", "Detection Time"])  

with tab_training:
	st.markdown('# Training Time Evaluation')
	st.dataframe(df_time_train)

with tab_prediction:
	st.markdown('# Selection Time Evaluation')
	df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_inf')
	scale = st.radio('Select scale', ['linear', 'log'], key='scale_prediction')
	plot_box_plot(df_toplot, measure_name='seconds', scale=scale)

with tab_inference:
	st.markdown('# Detection Time Evaluation')
	df_toplot = generate_dataframe(df_time, datasets, methods_family, length, type_exp='_time')
	scale = st.radio('Select scale', ['linear', 'log'], key='scale_inference')
	plot_box_plot(df_toplot, measure_name='seconds', scale=scale)
