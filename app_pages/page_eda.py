# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from src.utils import get_df

sns.set_style('whitegrid')


def page_eda_body():
    st.write("## Exploratory Data Analysis")
    blurb = "We looked at several facets of our dataset.\n"\
            "* We at the distribution of the features. We used a normality "\
            "test to see if any of the features were normally distributed.\n"\
            "* We looked at the correlation coefficients between different "\
            "features. We then focused on the correlation coefficients where "\
            "one feature was the home team.\n"\
            "* We looked at the Predictive Power score between the different "\
            "features. We then focused on their relation to our target: home "\
            "team wins.\n"
    st.write("We looked at several facets of our dataset. ")
    st.write("In progress...")
    st.write(blurb)
    data = get_df('game_data_clean','datasets/clean/csv')
    corr_df = get_df('eda_spearman_corr','datasets/clean/csv')
    features = list(corr_df.columns)
    feature_1 = st.selectbox("Feature 1", features, index=1)
    feature_2 = st.selectbox("Feature 2", features, index=2)
    feature1 = features[0]
    feature2 = features[1]
    chart_data = corr_df.filter([feature1,feature2])
    #plot = sns.scatterplot(data=data, x=feature1, y=feature2)
    fig, ax = plt.subplots(figsize=(10, 10)) 
    sns.countplot(data=data, x=feature1,
                       hue="home_wins",
                       order=data[feature1].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{feature1}", fontsize=20, y=1.05)
    st.pyplot(fig)

    st.write(f" The correlation between {feature1} and {feature2} is {corr_df[feature1][feature2]}")
    #st.pyplot(plot.fig)