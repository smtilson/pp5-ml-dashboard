# This file follows that of the Churnometer walkthrough project
import streamlit as st
from src.utils import disp


def page_conclusions_body():
    st.write("* [Business Requirements](#business-requirements)")
    st.write("* [Project Outcomes](#project-outcomes)")

    st.write("# Project Conclusions")
    st.write(
        "The project was a success as we were able to validate all of "
        "our hypohteses. We were able to deploy a predictor for games "
        "which the classification models have not yet seen."
    )
    st.write("## Business Requirements")
    st.success(
        "- **Business Requirement 1**: This requirement was met in two "
        "ways.  The features that have the highest correlation with "
        f"{disp('home_wins')} are (in order): \n"
        f"  * {disp('dreb_away')}\n"
        f"  * {disp('ast_home')}\n"
        f"  * {disp('dreb_home')}\n"
        f"  * {disp('ast_away')}\n"
        "Through training our classification models we found that REB "
        "Home and DREB Home were influential statistics.")
    st.success(
        "- **Business Requirement 2**: This requirement was met by "
        "training two models. Our Logistic Regression model has an "
        "average precision of 87.64% and an accuracy of 87.93%. Our "
        "Adaptive Boost model has an average precision of 86.97% and "
        "an accuracy of 87.44%."
    )
    st.success(
        "- **Business Requirement 3**: This requirement was met by "
        "clustering the data and examining the profiles of said "
        "clusters. The range of seasons for each cluster were "
        "concentrated (Q1-Q3) in non-overlapping ranges."
    )
    st.write("## Project Outcomes")
    st.info(
        "The models performed well. I would have liked to have done a more"
        " extensive hyperparameter search. I also would have liked to try "
        "different base estimators for the Adaptive Boost model, but this "
        "did not work. In the future, I would like to try and forecast "
        "wins by treating the data as a time series. I would also to look "
        "at higher dimensional representations of the data where "
        "individual teams and players are encoded. There is a lot more to "
        "explore in this data."
    )
