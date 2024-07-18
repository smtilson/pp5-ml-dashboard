# These functions are designed for plotting dataframes in Jupyter notebooks.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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