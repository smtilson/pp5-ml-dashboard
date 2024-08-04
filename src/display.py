# These functions are designed for displaying information in Jupyter notebooks
# and streamlit as tables or graphs/plots.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.model_eval import gen_clf_report
from src.utils import disp, undisp


def display_report(pipe, X, Y, label_map=None):
    matrix, performance, acc = gen_clf_report(X, Y, pipe, label_map)
    st.write("#### Confusion Matrix")
    st.dataframe(matrix)
    st.write("\n")
    st.write("#### Performance")
    st.write(f"Accuracy: {acc*100:.2f}%\n")
    st.dataframe(performance)


# Inspired by code from the Churnometer walkthrough
def display_features_logistic(pipe, X):
    coefficients = pipe["model"].coef_[0]
    try:
        initial_drop = pipe.steps[0][1].features_to_drop_
    except AttributeError as e:
        if "features_to_drop" in str(e):
            initial_drop = []
        else:
            raise e
    features = [disp(col) for col in X.columns if col not in initial_drop]
    importance_list = [
        (feature, X[undisp(feature)].std() * coeff)
        for feature, coeff in zip(features, coefficients)
    ]
    df_feature_importance = pd.DataFrame(
        data={
            "Features": [term[0] for term in importance_list],
            "Importance": [abs(term[1]) for term in importance_list],
        }
    ).sort_values(by="Importance", ascending=False)

    best_features = df_feature_importance["Features"].to_list()

    st.write(
        f"* These are the {len(best_features)} most important features in "
        "descending order. The model was trained on them: \n"
        f"{df_feature_importance['Features'].to_list()}"
    )
    st.write("\n")
    fig, axes = plt.subplots(figsize=(6, 2))
    sns.barplot(data=df_feature_importance, x="Features", y="Importance")
    plt.title("Feateure Importance")
    plt.xticks(rotation=70)
    st.pyplot(fig)
    st.dataframe(df_feature_importance)


# Inspired by code from the Churnometer walkthrough
def display_features_tree_based(pipe, X):
    try:
        initial_drop = pipe.steps[0][1].features_to_drop_
    except AttributeError as e:
        if "features_to_drop" in str(e):
            initial_drop = []
        else:
            raise e
    features = [disp(col) for col in X.columns if col not in initial_drop]
    df_feature_importance = pd.DataFrame(
        data={"Features": features,
              "Importance": pipe["model"].feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    best_features = df_feature_importance["Features"].to_list()
    st.write(
        f"* These are the {len(best_features)} most important features in "
        "descending order. The model was trained on them: \n"
        f"{df_feature_importance['Features'].to_list()}"
    )
    st.write("\n")
    fig, axes = plt.subplots(figsize=(6, 2))
    sns.barplot(data=df_feature_importance, x="Features", y="Importance")
    plt.title("Feateure Importance")
    plt.xticks(rotation=70)
    st.pyplot(fig)
    st.dataframe(df_feature_importance)
