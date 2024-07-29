import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from src.display import display_report
from src.utils import get_df


def page_logistic_model_body():

    st.write("## Logistic Regression Model")
    st.write("In progress...")
    st.write(
        "We performed an initial search to determine which type of algorithm we "
        "should use for our classification problem. We found that the Logistic "
        "Regression model performed well. Our target, from the business case, is "
        "(average) 75% precision and 70% accuracy. The Logistic regression with the default"
        " hyperparameters is average precision of 84.5% and an accuracy of 84%. Note "
        "that we did set the correlation threshold at 85%. This can be considered a hyperparameter,"
        " and in fact we treated it as one when tuning the model."
    )
    
    pipe_dir = f'outputs/ml_pipeline/predict_home_wins/'
    
    train_dir = 'datasets/train/csv'
    test_dir = 'datasets/test/csv'
    X_TrainSet = get_df('X_TrainSet', train_dir)
    Y_TrainSet = get_df('Y_TrainSet', train_dir)
    X_TestSet = get_df('X_TestSet', test_dir)
    Y_TestSet = get_df('Y_TestSet', test_dir)
    if st.button("Show initial report"):
        logistic_pipe_v0 = joblib.load(filename=pipe_dir+'v0/logistic_pipeline.pkl')
        st.write("## Initial Report\n")
        st.write("### Training Set\n")
        display_report(logistic_pipe_v0, X_TrainSet, Y_TrainSet)
        st.write("\n\n### Test Set\n")
        display_report(logistic_pipe_v0, X_TestSet, Y_TestSet)
        st.write("\nNotice that the performance on the training set and test set is similar."
                 " This is an indication that are model is generalizing well.")
    st.write("\n\nWe considered the following hyperparameters during tuning.")
    st.write("* `solver`: specifies the type of algorithm used")
    st.write("* `C`: determines the strength of the penalty function")
    st.write("* `penalty`: determines which penalty function will be used (not all penalty functions work for each solver).")
    st.write("* `threshold`: the cutoff threshold for feature selection based on correlation coefficient")
    st.write("After two grid searchs we settled on the following values for the above hyperparameters.")
    st.write("* correlation threshold: 0.77")
    st.write("* `C`: 500.5")
    st.write("* solver method: newton-cg")
    st.write("* penalty function: l2")
    st.write("After these two passes our models started to converge on certain precision and accuracy "
             "values. However, the number of different combinations of parameters with the same "
             "performance grew dramatically. This meant that it was harder to narrow down which "
             "combinations of parameters would improve the performance.")
    logistic_pipe_v1 = joblib.load(filename=pipe_dir+'v1/logistic_pipeline.pkl')
    st.write("## Final Report\n")
    st.write(" we were able to achieve a model that performed well above our business requirements.")
    st.write("\n### Training Set\n")
    display_report(logistic_pipe_v1, X_TrainSet, Y_TrainSet)
    st.write("\n### Test Set\n")
    display_report(logistic_pipe_v1, X_TestSet, Y_TestSet)
    st.write("We are quite happy with our model. It has an average precision of "
             "87.64% and an accuracy of 87.9%. During our search, it felt like "
             "we were approaching the limit of what the model was capable of. Please "
             "see our notebook Tuning Hyperparameters for more details.")

    # talk about type of model
    # talk about hyperparameters
    # talk about accuracy
    # talk about confusion matrix
    # talk about important features
    # talk about feature importance


