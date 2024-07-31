# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#from src.data_management import load_telco_data, load_pkl_file


def page_summary_body():
    st.write("## Summary")
    st.write("Fantasy sports is a huge industry. In 2022, the size of the "
             "market was [estimated](https://www.skyquestt.com/report/fantasy"
             "-sports-market) at $27 billion. We were approached by a "
             "fictional sports analyst with a focus on fantasy basketball. "
             "They asked us to study NBA statistics to determine patterns as "
             "well as new statistics that could be used to predict the "
             "winning team.")
    st.write("\n")
    st.write("## Business Requirements")
    st.info("- **Business Requirement 1**: The client has asked us for "
           "statistics associated with winning that are less obvious (not "
           "related to made shots).\n"
           "- **Business Requirement 2**: The client has asked for a model "
           'that can predict if the home team wins based on these "less '
           'obvious" statistics. The model will then be used to understand the'
           " impact/weight of these statistics. The model should have at "
           "least an avg precision of 75% and accuracy of 70%.\n"
           "- **Business Requirement 3**: The client has asked us for a "
           "clustering model in order to determine if meaningful trends can be"
           " detected by these methods.")
    st.write("\n")
    st.write("## Hypotheses")
    st.info("- **Hypothesis 1**: ML algorithms will gravitate towards the "
            "obvious statistics (point related, such as made shots) in order "
            "to construct their models.\n")
    st.warning("- **Validation**: train models on all features and determine "
               "which are selected most frequently.")
    st.info("- **Hypothesis 2**: There are features that don't involve made "
            "shots which correlate with winning.")
    st.warning("- **Validation**: correlation study analyzing the relationship"
               " between the above features and winning.")
    st.info("- **Hypothesis 3**: A good pipeline can be built based on "
            "features that don't include made shots.")
    st.warning("- **Validation**: drop made shots features and train a model "
               "achieving at least 75% avg. precision and 70% accuracy.")
    st.info("- **Hypothesis 4**: Eras of the NBA can be seen through their "
            "statistics.")
    st.warning("- **Validation**: train a clustering model with time removed "
               "and use a classification model with time added back to "
               "determine the profile of the clusters.")

    st.write("\n")
    st.write("## Results")
    st.write("We have validated all of our hypotheses. The jupyter notebooks "
             "in the repository can be followed to see our methods and "
             "approach. We have summarized our results and methods in this "
             "app. We have also provided some visualizations of our results "
             "and a predictor using our two final pipelines.")