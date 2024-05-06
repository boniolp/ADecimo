"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objs as go

from models.run_model import run_model
from src.pages.sidebar import show_sidebar
from src.utils.constant import list_measures, list_length, template_names, method_group
# from src.utils.helper import init_names, plot_box_plot
from src.pages.description_tab import show_description_tab



def main():
	# Setup
	st.set_page_config(page_title="ADecimoooooooo")

	# Loading data from CSV files
	df = pd.read_csv('data/merged_scores_{}.csv'.format('VUS_PR'))
	df = df.set_index('filename')

	df_time = pd.read_csv('data/inference_time.csv')
	df_time = df_time.rename(columns={'Unnamed: 0': 'filename'})
	df_time = df_time.set_index('filename')

	df_time_train = pd.read_csv('data/training_times.csv', index_col='window_size')
	# final_names = init_names(list_length, template_names)

	#plt.style.use('dark_background')
 
	# Sidebar
	metric_name, datasets, methods_family, length = show_sidebar(df, list_measures, method_group, list_length)

	# Reload the dataframe for scores based on the user-selected metric and re-index by filename
	df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
	df = df.set_index('filename')

	# Creating tabs in Streamlit for different sections of the app
	tab_desc, tab_acc, tab_time, tab_stats, tab_methods = st.tabs(["Description", "Accuracy", "Execution Time", "Datasets", "Methods"])  

	# Description Tab: Display a brief introduction and an image illustrating the model selection pipeline
	with tab_desc:
		show_description_tab()




if __name__ == "main":
	main()

exit()



# Accuracy Tab: This section is used for evaluating the accuracy of various classification algorithms
with tab_acc:
	st.markdown('# Accuracy Evaluation')
	st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark. Measure used: {}'.format(metric_name))
	tab_overall, tab_explore = st.tabs(["Overall results", "Explore the results"])  
	with tab_overall:
		df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_score')
		plot_box_plot(df_toplot, measure_name=metric_name)

	# Explore Results Tab: Allows for deeper exploration of data on a more granular level
	with tab_explore:
		col_dataset_exp, col_ts_exp, col_meth_exp, col_length_exp = st.columns([1, 1, 1, 1])
		with col_dataset_exp:
			dataset_exp = st.selectbox('Pick a dataset', list(set(df['dataset'].values))+['Upload your own'])
		with col_ts_exp:
			time_series_selected_exp = st.selectbox('Pick a time series', list(df.loc[df['dataset']==dataset_exp].index))
		with col_length_exp:
			length_selected_exp = st.selectbox('Pick a window length', list_length)
		with col_meth_exp:
			method_selected_exp = st.selectbox('Pick a method', [meth.format(length_selected_exp) for meth in methods_ens])
		
		# Handling file uploads for custom datasets
		if dataset_exp == 'Upload your own':
			uploaded_ts = st.file_uploader("Upload your time series")
			if uploaded_ts is not None:
				try:
					trace_scores_uplaod = []
					ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
					
					ts_data = ts_data_raw[:, 0].astype(float)
					ts_data = ts_data[:min(len(ts_data), 40000)]
					
					pred_detector, counter_dict = run_model(ts_data)
					st.markdown("Voting results:")
					st.bar_chart(counter_dict)
					st.markdown("The Detector to select is {}".format(pred_detector))
					
					
					
					trace_scores_uplaod.append(go.Scattergl(
						x=list(range(len(ts_data))), y=ts_data,
						xaxis='x', yaxis='y2', name = "Time series", mode = 'lines',
						line = dict(color = 'blue', width=3), opacity = 1
					))
					if len(ts_data_raw[0]) > 1:
						label_data = ts_data_raw[:, 1]
						label_data = label_data[:min(len(label_data), 40000)]
						anom = add_rect(label_data, ts_data)
						trace_scores_uplaod.append(go.Scattergl(
							x=list(range(len(ts_data))), y=anom,
							xaxis='x', yaxis='y2', name = "Anomalies",
							mode = 'lines', line = dict(color = 'red', width=3), opacity = 1
						))
					
					layout_upload = go.Layout(
						yaxis=dict(domain=[0, 0.4], range=[0, 1]),
						yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
						title="Uploaded time series snippet (40k points maximum)",
						template="simple_white",
						margin=dict(l=8, r=4, t=50, b=10),
						height=375,
						hovermode="x unified",
						xaxis=dict(range=[0, len(ts_data)])
					)

					fig = dict(data=trace_scores_uplaod, layout=layout_upload)
					st.plotly_chart(fig, use_container_width=True)


				except Exception as e:
					st.markdown('file format not supported yet, please upload a time series in the TSB-UAD format: {}'.format(e))
		else:
			path_ts = 'data/benchmark_ts/' + dataset_exp + '/' + time_series_selected_exp + '.zip'
			path_ts_score = {AD_method:'data/scores_ts/' + dataset_exp + '/' + AD_method + '/score/' + time_series_selected_exp + '.zip' for AD_method in old_method}
		
			st.markdown("Detector selected by {} : {}".format(method_selected_exp, df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')]))

			ts_data_raw = pd.read_csv(path_ts, compression='zip', header=None).dropna().to_numpy()
			label_data = ts_data_raw[:, 1]
			ts_data = ts_data_raw[:, 0].astype(float)
		

		

			score_AD_method = pd.DataFrame()
			for meth in path_ts_score.keys():
				score_AD_method[meth] = pd.read_csv(path_ts_score[meth], compression='zip', header=None).dropna().to_numpy()[:, 0].astype(float)

			#st.line_chart(ts_data)
			#st.area_chart(score_AD_method)

			anom = add_rect(label_data, ts_data)
			trace_scores = []
			trace_scores.append(go.Scattergl(
				x=list(range(len(ts_data))), y=ts_data,
				xaxis='x', yaxis='y2', name = "Time series", mode = 'lines',
				line = dict(color = 'blue', width=3), opacity = 1
			))
			trace_scores.append(go.Scattergl(
				x=list(range(len(ts_data))), y=anom,
				xaxis='x', yaxis='y2', name = "Anomalies",
				mode = 'lines', line = dict(color = 'red', width=3), opacity = 1
			))

			for method_name in score_AD_method.columns:
				if method_name == df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')]:
					alpha_val = 1
				else:
					alpha_val = 0.05
				trace_scores.append(go.Scattergl(
					x=list(range(len(ts_data))),
					y=[0] + list(score_AD_method[method_name].values[1:-1]) + [0],
					name = "{} score".format(method_name), opacity = alpha_val, mode = 'lines', fill="tozeroy",
				))

			layout = go.Layout(
				yaxis=dict(domain=[0, 0.4], range=[0, 1]),
				yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
				title="{} time series snippet (40k points maximum)".format(time_series_selected_exp),
				template="simple_white",
				margin=dict(l=8, r=4, t=50, b=10),
				height=375,
				hovermode="x unified",
				xaxis=dict(range=[0, len(ts_data)])
			)

			fig = dict(data=trace_scores, layout=layout)
			st.plotly_chart(fig, use_container_width=True)



with tab_time:
	st.markdown('# Execution Time Evaluation')
	st.markdown('Overall evaluation of 125 classification algorithm used for model selection for anoamly detection. We use the 496 randomly selected time series from the TSB-UAD benchmark.')
	tab_training, tab_prediction, tab_inference = st.tabs(["Training Time", "Selection Time", "Detection Time"])  
	with tab_training:
		st.markdown('## Training Time Evaluation')
		st.dataframe(df_time_train)
	with tab_prediction:
		st.markdown('## Selection Time Evaluation')
		df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_inf')
		scale = st.radio('Select scale', ['linear', 'log'], key='scale_prediction')
		plot_box_plot(df_toplot, measure_name='seconds', scale=scale)
	with tab_inference:
		st.markdown('## Detection Time Evaluation')
		df_toplot = generate_dataframe(df_time, datasets, methods_family, length, type_exp='_time')
		scale = st.radio('Select scale', ['linear', 'log'], key='scale_inference')
		plot_box_plot(df_toplot, measure_name='seconds', scale=scale)
		
	
with tab_stats:
	st.markdown('# Dataset Description')
	st.markdown(text_description_dataset)
	st.markdown('# Dataset Statistics')
	st.dataframe(df[dataset_stats])
	fig = plt.figure(figsize=(10, 20))
	for i, elem_stat in enumerate(dataset_stats_real):
		# plt.subplot(1, len(dataset_stats_real), 1+i)
		plt.subplot(4, 2, 1+i)
		sns.histplot(x=df[elem_stat].values, bins=30, fill=True)
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
	
	
