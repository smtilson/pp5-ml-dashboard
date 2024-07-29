# These functions are designed for plotting dataframes in Jupyter notebooks.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from model_eval import gen_clf_report


def heatmap_threshold(df,threshold, figsize=(8,8)):
    if len(df.columns) > 1:

      mask = np.zeros_like(df, dtype=np.bool)
      mask[abs(df) < threshold] = True

      fig, ax = plt.subplots(figsize=figsize)
      ax = sns.heatmap(df, annot=True, annot_kws={"size": 10},
                       mask=mask,cmap='rocket_r', linewidth=0.05,
                       linecolor='lightgrey')
      
      plt.ylim(len(df.columns),0)
      plt.show()

def display_report(pipe, X, Y):
    matrix, performance, acc = gen_clf_report(X, Y, pipe)
    st.write("#### Confusion Matrix\n")
    st.dataframe(matrix)
    st.write("#### Performance\n")
    st.write(f"Accuracy: {acc*100:.2f}%\n")
    st.dataframe(performance)