# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#from src.data_management import load_telco_data, load_pkl_file


def page_hypothesis_1_body():
    st.write("## Naive Feature Selection")
    st.write("Our first hypothesis, is that if allowed, all of our "
             "classification models will gravitate towards features that "
             "relate to made shots. These are:"
             "* `plus_minus_home (`plus_minus_away` was already removed)\n"
             "* `pts_home`, `pts_away`\n"
             "* `ftm_home`, `ftm_away`\n"
             "* `fgm_home`, `fgm_away` \n"
             "* `fg3m_home`, `fg3m_away` \n"
             "The first 2 rows are features from which the winner of the game "
             "can be directly determined. We saw in our correlation study that."
        "We did this to test Hypothesis 3: that a good pipeline can be made"
        " without these features.")