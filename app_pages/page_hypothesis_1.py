# This file follows that of the Churnometer walkthrough project
import streamlit as st


def page_hypothesis_1_body():
    # TOC
    st.write("* [Hypothesis 1](#hypothesis-1)")
    st.write("* [Process](#process)")
    st.write("* [Conclusions](#conclusion)")
    st.write("# Naive Feature Selection")
    # Introduction
    st.write("## Hypothesis")
    st.write(
        "Our first hypothesis relates to how models select features and "
        "which features the different supervised learning algorithms "
        "will prefer. The winner of a game of basketball is determined "
        "by the team that scores more points."
    )
    st.write("\n")
    st.info(
        "* Hypothesis 1: \n"
        "  * ML algorithms will gravitate towards the obvious statistics"
        " (point related, such as made shots) in order to construct "
        "their models."
    )
    st.write("\n")

    # Content
    st.write("## Process")
    st.write(
        "To validate this hypothesis, we looked trained various models "
        "on our data and looked at which features it used. We considered"
        " the following classification models:\n"
        "* Logistic Regression\n"
        "* Decision Tree Classifier\n"
        "* Random Forest Classifier\n"
        "* Gradient Boosting Classifier\n"
        "* Extra Trees Classifier\n"
        "* Adaptive Boost Classifier\n"
        "* XGBoost Classifier\n"
    )
    st.write(
        "Specifically, we attempted to validate Hypothesis 1 by doing the "
        "following:"
    )
    st.info(
        "Train the above models on all features and investigate which are"
        " selected. The details can be found in notebook 05."
    )
    st.write(
        "The 'point related' statistics are the following:\n"
        "* Plue Minus Home (`plus_minus_away` was already removed)\n"
        "* PTS, both Home and Away\n"
        "* FTM, both Home and Away\n"
        "* FGM, both Home and Away\n"
        "* FG3M, both Home and Away \n"
        "The first 2 rows are features from which the winner of the game "
        "can be directly determined. "
    )
    st.write(
        "Initially, most models only used plus/minus score. Half also "
        "used total points. All models performed at 100%, as expected. "
        "After dropping plus/minus score and refitting the pipelines, "
        "the models only used total points. Surprisingly, the performance"
        " of all models, except for the Logistic Regression model, "
        "dropped to 99.9%. Next, we dropped total points as well and "
        "refitted the pipelines. The worst performing model was the "
        "Decission Tree Classifier with avg precision of 87.5% and avg "
        "accuracy of 84.5%. While many models used attempted shots in "
        "addition to made shots, only the Extra Trees Classifier used a "
        "feature unrelated to shots, which was total rebounds."
    )
    st.write("## Conclusion")
    st.success(
        "This validates Hypothesis 1. At each stage, all models "
        "gravitated towards made shots. At stage one, all models used "
        "plus/minus score. At stage two, all models used total points."
        " At stage three, all models used made field goals. Thus, when "
        "possible, the models used features related to made shots."
    )
    st.info(
        "Interestingly, all models at stage three also used attempted free"
        " throws. This is likely due to the made free throws being dropped"
        " by the Smart Correlated Selection step of the pipeline. Only the"
        " Logistic Regression model used the 3 point shots. Similarly, "
        "half of the 3 point features were dorpped by the Smart "
        "Correlated Selection step of the pipeline."
    )
