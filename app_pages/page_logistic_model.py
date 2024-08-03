import streamlit as st
import joblib
from src.display import display_report, display_features_logistic
from src.utils import get_df


def page_logistic_model_body():
    # TOC
    st.write("* [Hyperparameters](#hyperparameters)\n")
    st.write("* [Performance Report](#performance-report)\n")
    st.write("* [Pipeline](#pipeline)\n")
    # Introduction
    st.write("# Logistic Regression Model")
    st.write(
        "During an initial search, we found that a Logistic Regression "
        "model based on performed well. Our target metrics, from the "
        "business case, is (avg) 75% precision and 70% accuracy. The "
        "Logistic Regression model with the default hyperparameters has "
        "average precision of 85.5% and an accuracy of 86%."
    )
    st.info(
        "Initially, we set the correlation threshold at 60%. We treated it as "
        "a hyperparameter when tuning the model."
    )
    st.info(
        " When training these models, we removed the features related to"
        " made shots and points. See the **ML: Naive Features Selection** page"
        " for details. We did this to test Hypothesis 3: that a good pipeline "
        "can be made without these features."
    )
    st.write("### Interesting findings")
    st.write(
        "Our Logistic Regression model found that the total rebounds "
        " were important features for this classification problem. It "
        "also found that this statistic for the home team was noticeably "
        "more important than that for the away team."
    )
    st.success("This satisfies our first Business requirement.")

    # Load Data
    pipe_dir = "outputs/ml_pipeline/predict_home_wins/v1/"
    logistic_pipe_v1 = joblib.load(filename=pipe_dir + "logistic_pipeline.pkl")
    train_dir = "datasets/train/classification"
    test_dir = "datasets/test/classification"
    X_TrainSet = get_df("X_TrainSet", train_dir)
    y_TrainSet = get_df("y_TrainSet", train_dir)
    X_TestSet = get_df("X_TestSet", test_dir)
    y_TestSet = get_df("y_TestSet", test_dir)

    # Hyperparameters
    st.write("## Hyperparameters")
    st.write("We considered the following hyperparameters during tuning.")
    st.write("* `solver = newton-cg`: specifies the type of algorithm used")
    st.write("* `C = 1000`: determines the strength of the penalty function")
    st.write(
        "* `penalty = l2`: determines which penalty function will be used (not"
        " all penalty functions work for each solver)."
    )
    st.write(
        "* `threshold = 0.81`: the cutoff threshold for feature selection "
        "based on correlation coefficient"
    )
    st.write(
        "After the second grid search, our models started to converge on "
        "certain precision and accuracy values. However, the number of "
        "different combinations of parameters with exactly the same "
        "performance grew dramatically. This meant that it was harder to "
        "narrow down which combinations of parameters would improve the "
        "performance. This is quite unlike the Adaptive boost model,"
    )

    # Report
    st.write("## Performance Report")
    st.write("Our Logistic Regression model exceed our business requirements."
             "Since the model performed similarly on the train and test data, "
             "our model is generalizing well.")
    st.write("### Training Set")
    display_report(logistic_pipe_v1, X_TrainSet, y_TrainSet)
    st.write("### Test Set")
    display_report(logistic_pipe_v1, X_TestSet, y_TestSet)
    st.success(
        "We are quite happy with our model. It has an average precision of "
        "87.64% and an accuracy of 87.93% on the test set. We feel that we "
        "were approaching the limit of what the model was capable of. Please "
        "see our notebook Tuning Hyperparameters for more details."
    )
    st.write("## Pipeline")
    st.write("### Important Features")
    st.info(
        "Using the (absolute values of the) coefficients or weights that the "
        "model attached to each feature, we are able to determine their "
        "relative importance."
    )
    st.write(
        "Below, we indicate the importance of the different features "
        "that the final model was trained on. This collection of "
        "features was determined by the `SmartCorrelatedSelection` "
        "from `feature_engine` and `FeatureSelection` from `sklearn`. "
        "The correlation threshold above is a parameter for "
        "`SmartCorrelationSelection`. This is in addition to the "
        "features we removed during the model selection process."
    )
    display_features_logistic(logistic_pipe_v1, X_TrainSet)
    st.write("We see that total rebounds of the home team are noticeably more "
             "impactful than total rebounds for the whole team.")
    st.write("### Pipeline Steps")
    st.write(logistic_pipe_v1)
