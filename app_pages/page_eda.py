# This file follows that of the Churnometer walkthrough project
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_df
from src.utils import disp, undisp

sns.set_style("whitegrid")


def page_eda_body():
    # TOC
    st.write("* [Feature Distributions](#feature-distributions)")
    st.write(
        "* [Correlation and Predictive Power score](#correlation-and-"
        "predictive-power-score)"
    )

    # Introduction
    st.write("# Exploratory Data Analysis")
    blurb = (
        "We looked at several facets of our dataset.\n"
        "* The distribution of the features: We used the "
        "[omnibus test of normality](https://docs.scipy.org/doc/scipy/"
        "reference/generated/scipy.stats.normaltest.html) "
        "to see if any of the features were normally distributed.\n"
        "* The correlation coefficients: We then focused on the correlation "
        "coefficients where one feature was the home team.\n"
        "* The Predictive Power score: We then focused on their relation to "
        "our target: home team wins."
    )
    st.write(blurb)
    data = get_df("game_eda", "datasets/clean/csv")
    corr_df = get_df("eda_spearman_corr", "eda")
    normality_scores = get_df("normality_scores", "eda")
    pps_results = get_df("pps_results", "eda")
    raw_features = [col for col in corr_df.columns]
    features = [disp(feature) for feature in raw_features]

    st.write("## Feature Distributions")
    st.write("### Normality Test")
    st.write("Select features to see their score on the normality test.")
    feature = st.selectbox("Feature", features, index=0, key="widget_1")
    feature = undisp(feature)
    score = round(normality_scores.loc[feature]["W"], 3)
    pval = normality_scores.loc[feature]["pval"]
    pval = rep_p_val(pval)
    st.write(f"{disp(feature)}: Score = {score}")
    st.write(f"{(2+len(disp(feature)))*' '}p-value =**{pval}**")
    st.write(
        "In order to be considered normal, the p-value must be larger than "
        "0.05."
    )
    st.write(
        "None of the features are normally distributed, but we were able to "
        "transform some of them in order to obtain normal distributions."
    )
    st.write("### Distributions")
    st.write("Select a feature to plot its distribution for home and away "
             "teams.")
    single_features = [disp(feature.split("_")[0]) for feature in raw_features]
    feature_dist = st.selectbox("Feature", single_features, index=0)
    home_feat = feature_dist.lower() + "_home"
    away_feat = feature_dist.lower() + "_away"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    home_score = rep_p_val(normality_scores.loc[home_feat]["pval"])
    away_score = rep_p_val(normality_scores.loc[away_feat]["pval"])
    sns.histplot(data=data, x=home_feat, kde=True, hue="home_wins", ax=axes[0])
    axes[0].set_title(f"\n{disp(home_feat)} p-value = {home_score}")
    sns.histplot(data=data, x=away_feat, kde=True, hue="home_wins", ax=axes[1])
    plt.title(
        f"{disp(feature_dist)} Distribution: Home vs. Away\n"
        f"{disp(away_feat)} p-value = {away_score}"
    )
    st.pyplot(fig)
    st.write(
        "Notice that there is a certain symmetry when interchanging the "
        "roles home and away. These distributions look normal but are not. We "
        "will be able to normalize some of them."
    )

    st.write("## Correlation and Predictive Power score\n")
    st.write(
        "Correlation (Spearman) coefficients are measures of the strength of "
        "the a monotonic relationship between two random variables. It is a "
        "very widely used measurement."
    )
    st.write(
        "Predictive Power score is an asymmetric statistic that "
        "can be used to analyze data. It can often find patterns that "
        "correlation coefficients miss."
    )
    st.write(
        "Select two features to compare. They will be plotted against "
        "each other in a scatter plot."
    )
    st.write("The features with the highest correlation coefficients are: \n")
    home_wins_index = raw_features.index("home_wins")
    pm_index = raw_features.index("plus_minus_home")
    feature_1 = st.selectbox(
        "Feature 1", features, index=home_wins_index, key="widget_2"
    )
    feature_2 = st.selectbox("Feature 2", features, index=pm_index,
                             key="widget_3")
    feature_1 = undisp(feature_1)
    feature_2 = undisp(feature_2)
    fig, axes = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=data, x=feature_1, y=feature_2, ax=axes)
    plt.title(
        f"{disp(feature_1)} vs. {disp(feature_2)}: "
        f"{corr_df[feature_1][feature_2]}"
    )
    st.pyplot(fig)
    st.write("These scatter plots help us understand if there is a "
             "relationship between the features.")
    ppscore = pps_results.query("x == @feature_1 & y == @feature_2")
    ppscore = round(ppscore["ppscore"].iloc[0], 4)
    st.write(
        "The respective Predictive Power scores are: \n"
        f"* {disp(feature_1)} influences {disp(feature_2)}: {ppscore}\n "
    )
    st.write("### Some interesting relationships:")
    st.write("* Year and 3 point shots (made as well as attempted)")
    st.write(
        "* Aside from point related statistics, the features that have "
        f"the highest correlation with {disp('home_wins')} are (in order): \n"
        f"  * {disp('dreb_away')}\n"
        f"  * {disp('ast_home')}\n"
        f"  * {disp('dreb_home')}\n"
        f"  * {disp('ast_away')}\n"
    )
    st.write(
        "* Only Plus/Minus Home and Pts Home have non-trivial predictive"
        " power scores with respect to Home Wins."
    )
    st.success("These relationships address Business Requirement 1 and "
               "Hytpothesis 2.")


def rep_p_val(pval: float) -> str:
    if 'e' in str(pval):
        val, exp = str(pval).split("e")
        val = round(float(val), 4)
        value = f"{val}e{exp}"
    else:
        value = str(pval)
    return value
