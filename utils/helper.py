"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from .constant import methods_colors, methods_conv, methods_sit, methods_ts, methods_classical, old_method, list_length, method_group, template_names


def init_names(list_length, template_names):
	"""
	Initializes and returns a dictionary with template names adjusted to the given list lengths.

	Args:
		list_length (list): List of lengths for which template names need adjustment.
		template_names (dict): Dictionary containing template names as keys.

	Returns:
		dict: A dictionary with adjusted template names based on the list lengths.

	Example:
		If list_length = [10, 20] and template_names = {'template_{}': 'value'}, 
		the function will return {'template_10': 'value10', 'template_20': 'value20'}.
	"""
	final_names = {}
	for length in list_length:
		for key, value in template_names.items():
			if '{}' in key:
				new_key = key.format(length)
				new_value = value.format(length)
				final_names[new_key] = new_value
			else:
				final_names[key] = value

	return final_names


def plot_box_plot(df, measure_name, scale='linear'):
	"""
	Plots a box plot using Seaborn and Matplotlib.

	Args:
		df (DataFrame): DataFrame containing data to be plotted.
		measure_name (str): Name of the measure for labeling the x-axis.
		scale (str, optional): Scale for the x-axis ('linear' or 'log'). Defaults to 'linear'.
	"""
	# Check if DataFrame is empty
	if df.empty:
		st.warning("👻 It's a ghost town in here... No data found for plotting! Please select something from above to view data.")
		return

	# Create color palette
	my_pal = {method: methods_colors["detectors"] for method in old_method}
	for family, color in zip([methods_conv, methods_sit, methods_ts, methods_classical],
							 [methods_colors["conv"], methods_colors["sit"], methods_colors["rocket"],
							  methods_colors["feature_based"]]):
		for length in list_length:
			my_pal_tmp = {method.format(length): color for method in family}
			my_pal = {**my_pal, **my_pal_tmp}
	my_pal.update({"Avg Ens": methods_colors["avg_ens"], 'Oracle': methods_colors["oracle"]})

	# Add same names with '_inf' instead of '_score'
	my_pal_plus = {}
	
	for key in my_pal.keys():
		my_pal_plus[key] = my_pal[key]
		my_pal_plus[key.replace('_score', '_inf')] = my_pal[key]
		my_pal_plus[key.replace('_score', '')] = my_pal[key]
		my_pal_plus[key.replace('_default', '')] = my_pal[key]
		my_pal_plus[key.replace('_default', '').replace('_score', '')] = my_pal[key]

	# # Melting dataframe to long format for better handling by Plotly Express
	# df_long = df.melt(var_name='Variable', value_name=measure_name)

	# # Determine order for box plot based on median values
	# order = {"Variable": list(df.median().sort_values().index)[::-1]}

	# # Create box plot using Plotly
	# fig = px.box(
	# 	df_long, 
	# 	x=measure_name, 
	# 	y='Variable',
	# 	category_orders=order,
	# 	color='Variable',
	# 	orientation='h',
	# 	title=f'Box Plot of {measure_name}',
	# 	color_discrete_map=my_pal_plus,
	# 	points='all',
	# )

	# # Adjust x-axis scale if specified
	# if scale == 'log':
	# 	fig.update_xaxes(type='log')

	# # Update layout and styles
	# fig.update_layout(
	# 	height=max(400, 30 * len(df.columns)),  # Dynamic height based on number of columns
	# 	width=800,  # Suitable width to handle large labels and readability
	# 	showlegend=False  # Optional: Hide legend if not necessary
	# )

	# # Customize hover data
	# fig.update_traces(hoverinfo='x+name')

	# # Display the plot in Streamlit
	# st.plotly_chart(fig, use_container_width=True)

	# Determine plot dimensions based on DataFrame length
	fig_height = min(30, max(1, int(0.40 * len(df.columns))))
	fig = plt.figure(figsize=(10, fig_height))

	# Determine order for box plot based on median values
	order = list(df.median().sort_values().index)[::-1]

	# Create box plot
	g = sns.boxplot(
		data=df, 
		order=order, 
		palette=my_pal_plus, 
		showfliers=False, 
		orient='h',
		saturation=1, 
		whis=0.241289844
	)
	
	g.xaxis.grid(True)
	plt.xlabel(measure_name)
	if scale == 'log':
		plt.xscale('log')
	st.pyplot(fig)



def generate_dataframe(df, datasets, methods_family, length, type_exp='_score'):
	"""
	Generates a DataFrame filtered by dataset, method families, and lengths, for different types of experiments.

	Args:
		df (DataFrame): Original DataFrame containing data to be filtered.
		datasets (list): List of datasets to include in the filtered DataFrame.
		methods_family (list): List of method families to include in the filtered DataFrame.
		length (list): List of lengths to include in the filtered DataFrame.
		type_exp (str, optional): Type of experiment ('_score', '_inf', or '_time'). Defaults to '_score'.

	Returns:
		DataFrame: Filtered DataFrame based on the specified parameters.

	Example:
		filtered_df = generate_dataframe(df, ['dataset1', 'dataset2'], ['family1', 'family2'], [10, 20], '_inf')
	"""

	# Filter DataFrame based on type of experiment
	if type_exp == '_score':
		# Generate DataFrame for scoring experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', type_exp)
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length] + old_method]
	elif type_exp == '_inf':
		# Generate DataFrame for inference experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', type_exp)
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length]]
	elif type_exp == '_time':
		# Generate DataFrame for time-related experiments
		return df.loc[df['dataset'].isin(datasets)][[method.format(l).replace('_score', '').replace('_default', '')
													  for method_g in methods_family
													  for method in method_group[method_g]
													  for l in length] + old_method]



def add_rect(label, data):
	"""
	Highlights anomalies in a time series plot.

	Args:
		label (list): List of labels indicating anomalies (1 for anomaly, 0 for normal).
		data (list): List of data points representing the time series.

	Returns:
		list: List containing highlighted anomalies in the time series plot.

	Example:
		highlighted_ts = add_rect([0, 1, 0, 0, 1], [1, 2, 3, 4, 5])
	"""

	# Initialize list for plotting anomalies
	anom_plt = [None] * len(data)

	# Create a copy of the original data
	ts_plt = data.copy()

	# Get the length of the time series
	len_ts = len(data)

	# Loop through labels and data to identify anomalies
	for i, lab in enumerate(label):
		if lab == 1:
			# Mark anomalies and neighboring points
			anom_plt[i] = data[i]
			anom_plt[min(len_ts - 1, i + 1)] = data[min(len_ts - 1, i + 1)]

	return anom_plt


def rename_columns(df):
	
	# Function to process each column name
	def process_column_name(col_name):
		for length in list_length:
			for template, nice_name in template_names.items():
				if col_name.startswith(template.format(length)):
					return col_name.replace(template.format(length), nice_name.format(length))
		# If no template matches, return the original column name
		return col_name

	# Rename columns using the function
	new_columns = {col: process_column_name(col) for col in df.columns}
	df.rename(columns=new_columns, inplace=True)
	return df


def rename_columns_and_update_palette(df, palette):
	# List of lengths used in renaming
	list_length = [16, 32, 64, 128, 256, 512, 768, 1024]

	# Function to process each column name and update the palette
	def process_column_name(col_name):
		for length in list_length:
			for template, nice_name in template_names.items():
				formatted_template = template.format(length)
				if col_name.startswith(formatted_template):
					new_name = col_name.replace(formatted_template, nice_name.format(length))
					if col_name in palette:
						palette[new_name] = palette.pop(col_name)
					return new_name
		return col_name  # Return original name if no template matches

	# Update columns and palette
	new_columns = {col: process_column_name(col) for col in df.columns}
	df.rename(columns=new_columns, inplace=True)

	return df, palette
