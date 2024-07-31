# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#from src.data_management import load_telco_data, load_pkl_file


def page_hypothesis_1_body():
    st.write("## Naive Feature Selection")
    st.write("Our first hypothesis relates to how models select features and "
             "which features the different supervised learning algorithms "
             "will prefer. The winner of a game of basketball is determined "
             "by the team that scores more points."
             )
    st.write("\n")
    st.info("* Hypothesis 1: \n"
             "  * ML algorithms will gravitate towards the obvious statistics"
             " (point related, such as made shots) in order to construct "
             "their models.")
    st.write("\n")
    st.write("To validate this hypothesis, we looked trained various models "
             "on our data and looked at which features it used. We considered"
             " the following classification models:\n"
             "* Logistic Regression\n"
             "* Decision Tree Classifier\n"
             "* Random Forest Classifier\n"
             "* Gradient Boosting Classifier\n"
             "* Extra Trees Classifier\n"
             "* Adaptive Boost Classifier\n"
             "* XGBoost Classifier\n")
    st.write("\n")
    st.write("Specifically, we attempted to validate Hypothesis 1 by doing "
             "the following:")
    st.info("Train the above models on all features and investigate which are"
            " selected.")
    st.write("\n")
    st.write("The 'point related' statistics are the following:\n"
             "* `plus_minus_home (`plus_minus_away` was already removed)\n"
             "* `pts_home`, `pts_away`\n"
             "* `ftm_home`, `ftm_away`\n"
             "* `fgm_home`, `fgm_away` \n"
             "* `fg3m_home`, `fg3m_away` \n"
             "The first 2 rows are features from which the winner of the game "
             "can be directly determined. We saw in our correlation study that"
    
        "We did this to test Hypothesis 3: that a good pipeline can be made"
        " without these features.")