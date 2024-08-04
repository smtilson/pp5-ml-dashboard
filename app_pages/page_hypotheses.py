# This file follows that of the Churnometer walkthrough project
import streamlit as st


def page_hypotheses_body():

    # TOC
    st.write("* [Business Requirements](#business-requirements)")
    st.write("* [Hypotheses](#hypotheses)")
    st.write("* [Results](#results)")

    # Content
    st.write("# Business Requirements and Hypotheses")
    st.write("## Business Requirements")
    st.info(
        "- **Business Requirement 1**: The client has asked us for "
        "statistics associated with winning that are less obvious (points "
        "and made shots).\n"
        "- **Business Requirement 2**: The client has asked for a model "
        'that can predict if the home team wins based on these "less '
        'obvious" statistics. The model will then be used to understand the'
        " impact/weight of these statistics. The model should have at "
        "least an avg precision of 75% and accuracy of 70%.\n"
        "- **Business Requirement 3**: The client has asked us for a "
        "clustering model in order to determine if meaningful trends can be"
        " detected by these methods."
    )
    st.write("\n")
    st.write("## Hypotheses")
    st.info(
        "- **Hypothesis 1**: ML algorithms will gravitate towards the "
        "obvious statistics (points and made shots) in order "
        "to construct their models.\n"
    )
    st.warning(
        "- **Validation Method**: train models on all features and determine "
        "which are selected most frequently."
    )
    st.write("This is validated on the **ML: Naive Feature Selection** page.")
    st.info(
        "- **Hypothesis 2**: There are features that don't involve made "
        "shots which correlate with winning."
    )
    st.warning(
        "- **Validation Method**: correlation study analyzing the relationship"
        " between the above features and winning."
    )
    st.write("This is validated on the **Exploratory Data Analysis** page.")
    st.info(
        "- **Hypothesis 3**: A good pipeline can be built based on "
        "features that don't include made shots."
    )
    st.warning(
        "- **Validation Method**: drop made shots features and train a model "
        "achieving at least 75% avg. precision and 70% accuracy."
    )
    st.write(
        "This is validated on the **ML: Logistic Regression Model** and "
        "**ML: Adaptive Boost Model** pages."
    )
    st.info(
        "- **Hypothesis 4**: Eras of the NBA can be seen through their "
        "statistics."
    )
    st.warning(
        "- **Validation Method**: train a clustering model with time removed "
        "and use a classification model with time added back to "
        "determine the profile of the clusters."
    )
    st.write("This is validated on the **ML: Cluster Analysis** page.")

    st.write("## Results")
    st.success(
        "We have validated all of our hypotheses. The jupyter notebooks "
        "in the repository can be followed to see our methods and "
        "more detailed results. We have summarized our results and "
        "methods in this app. We have also provided some visualizations "
        "of our results and a predictor using our two final pipelines."
    )
    st.info("[Here](https://github.com/smtilson/pp5-ml-dashboard) is a link "
            "to the github repository.")
