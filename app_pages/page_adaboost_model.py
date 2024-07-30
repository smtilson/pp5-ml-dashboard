import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from src.display import display_report, display_features_tree_based
from src.utils import get_df


def page_adaboost_model_body():
    # Introduction
    st.write("## Adaptive Boost Model")
    st.write(
        "During an initial search, we found that an Adaptive Boost model "
        "based on Decision Trees performed well. Our target metrics, from the"
        " business case, is (avg) 75% precision and 70% accuracy. The Adaptive"
        " Boost model with the default hyperparameters has average precision "
        "of 82.5% and an accuracy of 82%."
    )
    st.write("\n")
    st.write(
        "Note that we set the correlation threshold at 60%. We treated this as a "
        " hyperparameter when tuning the model."
    )
    st.write("### Interesting findings")
    st.write(
        "Our Adaptive boost model found that the defensive rebounds "
        " were important features for this classification problem. It "
        "also found that this statistic for the home team was noticeably "
        "more important than that for the away team."
    )

    # Load Data
    pipe_dir = f"outputs/ml_pipeline/predict_home_wins/"
    ada_pipe_v1 = joblib.load(filename=pipe_dir + "v1/ada_pipeline.pkl")
    train_dir = "datasets/train/csv"
    test_dir = "datasets/test/csv"
    X_TrainSet = get_df("X_TrainSet", train_dir)
    Y_TrainSet = get_df("Y_TrainSet", train_dir)
    X_TestSet = get_df("X_TestSet", test_dir)
    Y_TestSet = get_df("Y_TestSet", test_dir)

    # Hyperparameters
    st.write("\n")
    st.write("### Hyperparameters")
    st.write("We considered the following hyperparameters during tuning.")
    st.write("* `algorithm = SAMME.R`: specifies the algorithm used")
    st.write("* `learning_rate = 1.133`: determines the weights for the " "estimators")
    st.write("* `n_estimators = 110`: the number of Decision Trees used.")
    st.write(
        "* `threshold = 0.8`: the cutoff threshold for feature selection "
        "based on correlation coefficient"
    )
    st.write(
        "Unlike the Logistic Regression model, the hyperparameters settled "
        "into a few ranges that we were able to explore in more depth. After "
        "five grid searches, we settled on the above values. The `SAMME.R` "
        "algorithm was quickly decided upon. There is more room for "
        "exploration with other hyperparameters and different base learners."
    )

    # Report
    st.write("\n")
    st.write("## Performance Report")
    st.write("Our Adaptive Boost model exceed our business requirements.")
    st.write("\n")
    st.write("### Training Set")
    st.write("\n")
    display_report(ada_pipe_v1, X_TrainSet, Y_TrainSet)
    st.write("\n\n")
    st.write("### Test Set")
    st.write("\n")
    display_report(ada_pipe_v1, X_TestSet, Y_TestSet)
    st.write(
        "We are quite happy with our model. It has an average precision of "
        "86.97% and an accuracy of 87.44%. During our search, it felt like "
        "we were approaching the limit of what the model was capable of. Please "
        "see our notebook Tuning Hyperparameters for more details."
    )
    st.write("\n\n")
    st.write("## Important Features")
    st.writw(
        "Tree based models come with an `feature_importance_` attribute "
        "that we can access after they have been trained."
    )
    st.write("\n")
    st.write(
        "Below, we indicate the importance of the different features "
        "that the final model was trained on. This collection of "
        "features was determined by the `SmartCorrelatedSelection` "
        "from `feature_engine` and `FeatureSelection` from `sklearn`. "
        "The correlation threshold above is a parameter for "
        "`SmartCorrelationSelection`. This is in addition to the "
        "features we removed during the model selection process."
    )
    display_features_tree_based(ada_pipe_v1, X_TrainSet)
