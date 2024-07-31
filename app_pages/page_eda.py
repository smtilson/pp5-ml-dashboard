# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from src.utils import get_df
from src.utils import display_feature_name as disp

sns.set_style("whitegrid")


def page_eda_body():
    st.write("## Exploratory Data Analysis")
    blurb = (
        "We looked at several facets of our dataset.\n"
        "* We looked at the distribution of the features. We used the "
        "[omnibus test of normality](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html) "
        "to see if any of the features were normally distributed.\n"
        "* We looked at the correlation coefficients between different "
        "features. We then focused on the correlation coefficients where "
        "one feature was the home team.\n"
        "* We looked at the Predictive Power score between the different "
        "features. We then focused on their relation to our target: home "
        "team wins.\n"
    )
    st.write(blurb)
    data = get_df("game_eda", "datasets/clean/csv")
    corr_df = get_df("eda_spearman_corr", "eda")
    normality_scores = get_df("normality_scores", "eda")
    pps_results = get_df("pps_results", "eda")
    features = list(corr_df.columns)

    st.write("## Feature Distributions")
    st.write("Select a feature to see what its score on the normality test " "is.")
    feature = st.selectbox("Feature", features, index=0)
    score = round(normality_scores.loc[feature]["W"], 3)
    pval = round(normality_scores.loc[feature]["pval"], 3)
    st.write(
        f"{disp(feature)} scored {score} and has a p-value f {pval}. In "
        "order to be considered normal, the p-value must be larger than "
        "0.05."
    )
    st.write(
        "While none of the features were found to be normally "
        "distributed, we were able to transform some of them in order to "
        "obtain normal distributions."
    )
    st.write("Select a feature to plot its distribution for home and away " "teams.")
    single_features = [feature.split("_")[0] for feature in features]
    feature_dist = st.selectbox("Feature", single_features, index=0)
    home_feat = feature_dist + "_home"
    away_feat = feature_dist + "_away"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    home_score = round(normality_scores.loc[home_feat]["pval"], 4)
    away_score = round(normality_scores.loc[away_feat]["pval"], 4)
    sns.histplot(data=data, x=home_feat, kde=True, hue="home_wins", ax=axes[0])
    axes[0].set_title(f"\n{disp(home_feat)} p-value = {home_score}")
    sns.histplot(data=data, x=away_feat, kde=True, hue="home_wins", ax=axes[1])
    plt.title(
        f"{disp(feature_dist)} Distribution: Home vs. Away\n"
        f"{disp(away_feat)} p-value = {away_score}"
    )
    st.pyplot(fig)
    st.write(
        "In order for the test to determine that the distribution is "
        "normal, we must have a p-value larger than 0.05."
    )
    st.write(
        "Notice that there is a certain symmetry when interchanging the "
        "roles home and away."
    )

    st.write("## Correlation (Spearman) and Predictive Power score\n")
    st.write(
        "Correlation coefficients are measures of the strength of the "
        "relationship between two random variables. It is a very widely "
        "used measurement."
    )
    st.write(
        "The Predictive Power score is an asymmetric statistic that "
        "can be used to analyze data. It can often find patterns that "
        "correlation coefficients miss."
    )
    st.write(
        "Select two features to compare. they will be plotted against "
        "each other in a scatter plot."
    )
    st.write("The features with the highest correlation coefficients are: \n")
    home_wins_index = features.index("home_wins")
    pm_index = features.index("plus_minus_home")
    feature_1 = st.selectbox("Feature 1", features, index=home_wins_index)
    feature_2 = st.selectbox("Feature 2", features, index=pm_index)
    fig, axes = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=data, x=feature_1, y=feature_2, ax=axes)
    plt.title(
        f"{disp(feature_1)} vs. {disp(feature_2)}: " f"{corr_df[feature_1][feature_2]}"
    )
    st.pyplot(fig)
    ppscore = pps_results.query("x == @feature_1 & y == @feature_2")
    ppscore = round(ppscore["ppscore"].iloc[0], 4)
    st.write(
        "The respective Predictive Power scores are: \n"
        f"* {disp(feature_1)} influences {disp(feature_2)}: {ppscore}\n "
    )

    st.write("### Some interesting relationships:")
    st.write("* Year and 3 point shots (made as well as attempted)")
    st.write(
        "* Aside from point related statistics, the features that had "
        f"the highest correlation with {disp('home_wins')} are (in order): \n"
        f"  * {disp('ast_away')}\n"
        f"  * {disp('ast_home')}\n"
        f"  * {disp('dreb_away')}\n"
        f"  * {disp('dreb_home')}\n"
    )
    st.write(
        "* Only Plus/Minus Home and Pts Home have non-trivial predictive"
        " power scores with respect to Home Wins."
    )
    st.write("\n")
    st.write("These relationships address Business Requirement 1 and "
             "Hytpothesis 2.")
