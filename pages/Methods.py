"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""


import streamlit as st

from utils.constant import text_description_AD, text_description_MS



tab_AD, tab_MS = st.tabs(["Anomaly Detection Methods", "Model Selection Methods"])
with tab_AD:
	st.markdown("# Anomaly Detection Methods")
	st.markdown(text_description_AD)
	
with tab_MS:
	st.markdown("# Model Selection Methods")
	st.markdown(text_description_MS)
