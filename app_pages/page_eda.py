# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from src.utils import get_df

sns.set_style('whitegrid')


def page_eda_body():
    data = get_df('game_data_clean','datasets/clean/csv')
    
    pass