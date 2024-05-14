"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from utils.constant import list_measures, list_length, method_group, methods_ens, old_method, all_datasets
from utils.helper import generate_dataframe, plot_box_plot, add_rect
from models.run_model import run_model



# Page Configuration and Title
st.markdown('# Accuracy Evaluation')
st.markdown('Overall evaluation of 125 classification algorithms used for model selection for anomaly detection. '
			'We utilize 496 randomly selected time series from the TSB-UAD benchmark.')

# Create tabs for displaying results
tab_overall, tab_explore = st.tabs(["Overall results", "Explore the results"])

# Tab for overall results with inline selection
with tab_overall:
	# Setup columns for selecting metric, dataset, method, and window length
	col_metric_over, col_dataset_over, col_method_over, col_length_over = st.columns([1, 1, 1, 1])
	
	# Metric selection
	with col_metric_over:
		metric_name = st.selectbox('Pick a measure', 
								   list_measures, 
								   help="Select the accuracy metric to evaluate the models.")

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

	# Loading data from CSV files
	df = pd.read_csv('data/merged_scores_{}.csv'.format(metric_name))
	df = df.set_index('filename')

	# Generate dataframe for plotting
	df_toplot = generate_dataframe(df, datasets, methods_family, length, type_exp='_score')
	st.dataframe(df_toplot)

	# Plot box plot using Plotly
	plot_box_plot(df_toplot, measure_name=metric_name)

# Tab for exploring individual results
with tab_explore:
	# Setup columns for selecting dataset, time series, method, and window length
	col_dataset_exp, col_ts_exp, col_meth_exp, col_length_exp = st.columns([1, 1, 1, 1])
	
	 # Dataset selection, including an option to upload custom dataset
	with col_dataset_exp:
		dataset_exp = st.selectbox('Pick a dataset', all_datasets + ['Upload your own'])
	
	# Time series selection based on the chosen dataset
	with col_ts_exp:
		time_series_selected_exp = st.selectbox('Pick a time series', list(df.loc[df['dataset'] == dataset_exp].index))
	
	# Window length selection
	with col_length_exp:
		length_selected_exp = st.selectbox('Pick a window length', list_length)
	
	# Method selection, showing methods tailored to selected window length
	with col_meth_exp:
		method_selected_exp = st.selectbox('Pick a method', [meth.format(length_selected_exp) for meth in methods_ens])
  	
	# Custom dataset upload handling
	if dataset_exp == 'Upload your own':
		uploaded_ts = st.file_uploader("Upload your time series")
		if uploaded_ts:
			try:
				# Process uploaded time series data
				ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
				ts_data = ts_data_raw[:, 0].astype(float)
				ts_data = ts_data[:min(len(ts_data), 40000)]  # Limit data points to improve performance
				
				# Run model on the uploaded data
				pred_detector, counter_dict = run_model(ts_data)
				st.markdown("Voting results:")
				st.bar_chart(counter_dict)
				st.markdown(f"The Detector to select is {pred_detector}")
				
				# Plot time series and detected anomalies
				trace_scores_upload = [go.Scattergl(x=list(range(len(ts_data))), y=ts_data,
													mode='lines', line=dict(color='blue', width=3),
													name="Time series", yaxis='y2')]
				if len(ts_data_raw[0]) > 1:
					label_data = ts_data_raw[:, 1]
					label_data = label_data[:min(len(label_data), 40000)]
					anom = add_rect(label_data, ts_data)
					trace_scores_upload.append(go.Scattergl(x=list(range(len(ts_data))), y=anom,
															mode='lines', line=dict(color='red', width=3),
															name="Anomalies", yaxis='y2'))
				
				# Define layout for uploaded data plot
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
				
				# Create and display the plot
				fig = dict(data=trace_scores_upload, layout=layout_upload)
				st.plotly_chart(fig, use_container_width=True)
			except Exception as e:
				st.error(f'File format not supported yet, please upload a time series in the TSB-UAD format: {e}')
	else:
		# Load data for the selected dataset and time series
		path_ts = f'data/benchmark_ts/{dataset_exp}/{time_series_selected_exp}.zip'
		path_ts_score = {AD_method: f'data/scores_ts/{dataset_exp}/{AD_method}/score/{time_series_selected_exp}.zip' for AD_method in old_method}
		
		# Display the detector selected by the chosen method
		st.markdown(f"Detector selected by {method_selected_exp}: {df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')]}")
		
		# Load time series and anomaly data
		ts_data_raw = pd.read_csv(path_ts, compression='zip', header=None).dropna().to_numpy()
		label_data = ts_data_raw[:, 1]
		ts_data = ts_data_raw[:, 0].astype(float)

		# Load scores for different methods and plot them
		score_AD_method = pd.DataFrame()
		for meth in path_ts_score.keys():
			score_AD_method[meth] = pd.read_csv(path_ts_score[meth], compression='zip', header=None).dropna().to_numpy()[:, 0].astype(float)

		# Prepare and display plots for the data and scores
		trace_scores = [go.Scattergl(x=list(range(len(ts_data))), y=ts_data, mode='lines', line=dict(color='blue', width=3), name="Time series", yaxis='y2'),
						go.Scattergl(x=list(range(len(ts_data))), y=add_rect(label_data, ts_data), mode='lines', line=dict(color='red', width=3), name="Anomalies", yaxis='y2')]
		for method_name in score_AD_method.columns:
			alpha_val = 1 if method_name == df.at[time_series_selected_exp, method_selected_exp.replace('_score', '_class')] else 0.05
			trace_scores.append(go.Scattergl(x=list(range(len(ts_data))), y=[0] + list(score_AD_method[method_name].values[1:-1]) + [0],
											 mode='lines', fill="tozeroy", name=f"{method_name} score", opacity=alpha_val))

		# Define layout for the plot
		layout = go.Layout(
			yaxis=dict(domain=[0, 0.4], range=[0, 1]),
			yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
			title=f"{time_series_selected_exp} time series snippet (40k points maximum)",
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(range=[0, len(ts_data)])
		)
		
		# Create and display the plot
		fig = dict(data=trace_scores, layout=layout)
		st.plotly_chart(fig, use_container_width=True)

