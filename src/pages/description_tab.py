
"""
@who: Paul Boniol, Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: Sylligardos PhD, 1st year (2024)
@what: ADecimo
"""

import streamlit as st
from PIL import Image

from ..utils.constant import description_intro


def show_description_tab():
	st.markdown(description_intro)
	image = Image.open('figures/3_pipeline.jpg')
	st.image(image, caption='Overview of the model selection pipeline')
