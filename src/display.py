# These functions are designed for plotting dataframes in Jupyter notebooks.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.model_eval import gen_clf_report
from src.notebook_functions import find_features


def heatmap_threshold(df, threshold, figsize=(8, 8)):
    if len(df.columns) > 1:

        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True

        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(
            df,
            annot=True,
            annot_kws={"size": 10},
            mask=mask,
            cmap="rocket_r",
            linewidth=0.05,
            linecolor="lightgrey",
        )

        plt.ylim(len(df.columns), 0)
        plt.show()


def display_report(pipe, X, Y):
    matrix, performance, acc = gen_clf_report(X, Y, pipe)
    st.write("#### Confusion Matrix\n")
    st.dataframe(matrix)
    st.write("#### Performance\n")
    st.write(f"Accuracy: {acc*100:.2f}%\n")
    st.dataframe(performance)


def display_features_logistic(pipe, X):
    coefficients = pipe['model'].coef_[0]
    initial_drop = pipe.steps[0][1].features_to_drop_
    features = [col for col in X.columns if col not in initial_drop]
    importance_list = [
        (feature, X[feature].std() * coeff)
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
    fig, axes = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=df_feature_importance, x="Features", y="Importance")
    plt.title("Feateure Importance")
    plt.xticks(rotation=70)
    st.pyplot(fig)
    st.dataframe(df_feature_importance)
    

def display_features_tree_based(pipe, X):
    initial_drop = pipe.steps[0][1].features_to_drop_
    features = [col for col in X.columns if col not in initial_drop]
    df_feature_importance = pd.DataFrame(
        data={"Features": features, 
              "Importance": pipe['model'].feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    best_features = df_feature_importance["Features"].to_list()
    st.write(
        f"* These are the {len(best_features)} most important features in "
        "descending order. The model was trained on them: \n"
        f"{df_feature_importance['Features'].to_list()}"
    )
    fig, axes = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=df_feature_importance, x="Features", y="Importance")
    plt.title("Feateure Importance")
    plt.xticks(rotation=70)
    st.pyplot(fig)
    st.dataframe(df_feature_importance)
