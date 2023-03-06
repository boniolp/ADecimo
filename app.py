from collections import namedtuple
import altair as alt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objs as go

from constant import *


#plt.style.use('dark_background')


df = pd.read_csv('data/merged_scores_{}.csv'.format('VUS_PR'))
df = df.set_index('filename')

df_time = pd.read_csv('data/inference_time.csv')
df_time = df_time.rename(columns={'Unnamed: 0': 'filename'})
df_time = df_time.set_index('filename')

df_time_train = pd.read_csv('data/training_times.csv')

def plot_box_plot(df,measure_name,scale='linear'):
	if len(df.columns) > 0:
		st.dataframe(df_toplot)
		fig = plt.figure(figsize=(10, min(30,max(1,int(0.40*len(df.columns))))))
		order = list(df_toplot.median().sort_values().index)[::-1]
		sns.boxplot(data=df_toplot,order=order,showfliers = False, orient='h') 
		plt.xlabel(measure_name)
		if scale == 'log':
			plt.xscale('log')
		st.pyplot(fig)
		
def generate_dataframe(df,datasets,methods_family,length,type_exp='_score'):
	if type_exp == '_score':
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score',type_exp) for method_g in methods_family for method in method_group[method_g] for l in length]+old_method]
	elif type_exp == '_inf':
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score',type_exp) for method_g in methods_family for method in method_group[method_g] for l in length]]
	elif type_exp == '_time':
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score','').replace('_default','') for method_g in methods_family for method in method_group[method_g] for l in length]+old_method]
		
		
def add_rect(label,data):
	anom_plt = [None]*len(data)
	ts_plt = data.copy()
	len_ts = len(data)
	for i,lab in enumerate(label):
		if lab == 1:
			anom_plt[i] = data[i]
			anom_plt[min(len_ts-1,i+1)] = data[min(len_ts-1,i+1)]
	return anom_plt
	
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
	if all_method: methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()), list(method_group.keys()),key='selector_methods_all')
	else: methods_family = container_method.multiselect('Select a group of methods', list(method_group.keys()),key='selector_methods')
	
	container_length = st.container()
	all_length = st.checkbox("Select all",key='all_length')
	if all_length: length = container_length.multiselect('Select a window length', list_length, list_length,key='selector_length_all')
	else: length = container_length.multiselect('Select a window length', list_length,key='selector_length')

df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
df = df.set_index('filename')

tab_desc,tab_acc, tab_time, tab_stats, tab_methods = st.tabs(["Description","Accuracy", "Execution Time", "Datasets", "Methods"])  
with tab_desc:
	st.markdown(description_intro)
	image = Image.open('figures/pipeline.jpg')
	st.image(image, caption='Overview of the model selection pipeline')
with tab_acc:
	st.markdown('# Accuracy Evaluation')
	st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark. Measure used: {}'.format(metric_name))
	tab_overall, tab_explore = st.tabs(["Overall results", "Explore the results"])  
	with tab_overall:
		df_toplot = generate_dataframe(df,datasets,methods_family,length,type_exp='_score')
		plot_box_plot(df_toplot,measure_name=metric_name)
	with tab_explore:
		col_dataset_exp, col_ts_exp, col_meth_exp, col_length_exp = st.columns([1,1,1,1])
		with col_dataset_exp:
			dataset_exp = st.selectbox('Pick a dataset', list(set(df['dataset'].values)))
		with col_ts_exp:
			time_series_selected_exp = st.selectbox('Pick a time series', list(df.loc[df['dataset']==dataset_exp].index))
		with col_length_exp:
			length_selected_exp = st.selectbox('Pick a window length', list_length)
		with col_meth_exp:
			method_selected_exp = st.selectbox('Pick a method', [meth.format(length_selected_exp) for meth in methods_ens])
		
		path_ts = 'data/benchmark_ts/' + dataset_exp + '/' + time_series_selected_exp + '.zip'
		path_ts_score = {AD_method:'data/scores_ts/' + dataset_exp + '/' + AD_method + '/score/' + time_series_selected_exp + '.zip' for AD_method in old_method}
		
		st.markdown("Detector selected by {} : {}".format(method_selected_exp,df.at[time_series_selected_exp,method_selected_exp.replace('_score','_class')]))

		ts_data_raw = pd.read_csv(path_ts,compression='zip', header=None).dropna().to_numpy()
		label_data = ts_data_raw[:,1]
		ts_data = ts_data_raw[:,0].astype(float)
		

		

		score_AD_method = pd.DataFrame()
		for meth in path_ts_score.keys():
			score_AD_method[meth] = pd.read_csv(path_ts_score[meth],compression='zip', header=None).dropna().to_numpy()[:,0].astype(float)
		
		#st.line_chart(ts_data)
		#st.area_chart(score_AD_method)

		anom = add_rect(label_data,ts_data)
		trace_scores = []
		trace_scores.append(go.Scattergl(
			x=list(range(len(ts_data))),y=ts_data,
			xaxis='x',yaxis='y2',name = "Time series",mode = 'lines',
			line = dict(color = 'blue',width=3),opacity = 1
		))
		trace_scores.append(go.Scattergl(
			x=list(range(len(ts_data))),y=anom,
			xaxis='x',yaxis='y2',name = "Anomalies",
			mode = 'lines',line = dict(color = 'red',width=3),opacity = 1
		))

		for method_name in score_AD_method.columns:
			if method_name == df.at[time_series_selected_exp,method_selected_exp.replace('_score','_class')]:
				alpha_val = 1
			else:
				alpha_val = 0.2
			trace_scores.append(go.Scattergl(
				x=list(range(len(ts_data))),
				y=[0] + list(score_AD_method[method_name].values[1:-1]) + [0],
				name = "{} score".format(method_name),opacity = alpha_val,mode = 'lines',fill="tozeroy",
			))

		layout = go.Layout(
			yaxis=dict(domain=[0, 0.4],range=[0,1]),
			yaxis2=dict(domain=[0.45, 1],range=[min(ts_data),max(ts_data)]),
			title="{} time series snippet (40k points maximum)".format(time_series_selected_exp),
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(range=[0,len(ts_data)])
		)

		fig = dict(data=trace_scores, layout=layout)
		st.plotly_chart(fig, use_container_width=True)



with tab_time:
	st.markdown('# Execution Time Evaluation')
	st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark.')
	tab_training, tab_prediction, tab_inference = st.tabs(["Training Time", "Prediction Time", "Inference Time"])  
	with tab_training:
		st.markdown('## Training Time Evaluation')
		st.dataframe(df_time_train)
	with tab_prediction:
		st.markdown('## Prediction Time Evaluation')
		df_toplot = generate_dataframe(df,datasets,methods_family,length,type_exp='_inf')
		scale = st.radio('Select scale', ['linear', 'log'],key='scale_prediction')
		plot_box_plot(df_toplot,measure_name='seconds',scale=scale)
	with tab_inference:
		st.markdown('## Inference Time Evaluation')
		df_toplot = generate_dataframe(df_time,datasets,methods_family,length,type_exp='_time')
		scale = st.radio('Select scale', ['linear', 'log'],key='scale_inference')
		plot_box_plot(df_toplot,measure_name='seconds',scale=scale)
		
	
with tab_stats:
	st.markdown('# Dataset Description')
	st.markdown(text_description_dataset)
	st.markdown('# Dataset Statistics')
	st.dataframe(df[dataset_stats])
	fig = plt.figure(figsize=(20, 4))
	for i,elem_stat in enumerate(dataset_stats_real):
		plt.subplot(1,len(dataset_stats_real),1+i)
		sns.histplot(x=df[elem_stat].values,bins=30, fill=True)
		plt.xlabel(elem_stat)
		plt.yscale('log')
		plt.tight_layout()
	st.pyplot(fig)
	
with tab_methods:
	tab_AD, tab_MS = st.tabs(["Anomaly Detection Methods", "Model Selection Methods"])
	with tab_AD:
		st.markdown("# Anomaly Detection Methods")
		st.markdown(text_description_AD)
		
	with tab_MS:
		st.markdown("# Model Selection Methods")
		st.markdown(text_description_MS)
	
	
