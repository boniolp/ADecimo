"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constant import text_description_dataset, dataset_stats_real, dataset_stats, all_datasets, template_names



# Loading data from CSV files
df = pd.read_csv('data/merged_scores_{}.csv'.format("VUS_PR"))
df = df.set_index('filename')



# Create tabs for displaying results
tab_description, tab_statistics = st.tabs(["Dataset Description", "Dataset Statistics"])

# Tab for overall results with inline selection
with tab_description:
	st.markdown('# Dataset Description')
	st.markdown(text_description_dataset)


with tab_statistics:
	# Setup columns for selecting dataset
	# st.columns returns a list of column objects, you need to specify which column to use
	col_dataset_over = st.columns(1)[0]  # Get the first (and only) column object from the list
	
	# Dataset selection
	with col_dataset_over:
		datasets = st.multiselect('Pick datasets', 
								  all_datasets, 
								  help="Select one or more datasets for analysis.")
	
	st.markdown('# Dataset Statistics')
	if datasets:
		st.dataframe(df[df['dataset'].isin(datasets)][dataset_stats_real])
		print(dataset_stats_real)
		# Filter the dataframe based on selected datasets
		filtered_df = df[df['dataset'].isin(datasets)]
		
		# Loop over the statistical measures and create individual histograms
		for elem_stat in dataset_stats_real:
			# Create a new figure for each statistic
			fig = go.Figure()

			for idx_dataset, dataset in enumerate(datasets):
				# Filter data for each dataset
				dataset_df = filtered_df[filtered_df['dataset'] == dataset]
				
				# Create histogram trace for each dataset
				trace = go.Histogram(
					x=dataset_df[elem_stat],
					name=dataset,
				)
				
				# Add trace to the figure
				fig.add_trace(trace)

			# Customize the layout
			fig.update_layout(
				title=f'Distribution of {template_names[elem_stat]}',
				xaxis_title=template_names[elem_stat],
				# yaxis_type='log',  # Set y-axis to logarithmic scale
				legend_title='Dataset',
				# height=400,  # Adjust height to fit the content
				# margin=dict(l=20, r=20, t=50, b=20)
			)

			# Show the plot in Streamlit
			st.plotly_chart(fig, use_container_width=True)

	else:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")