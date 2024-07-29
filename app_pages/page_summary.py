# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
#from src.data_management import load_telco_data, load_pkl_file


def page_summary_body():
    st.write("## Summary")
    st.write("Americans wagered more than $100 billion on sports in 2023. The"
             " market is continuing to grow as legal outlets for sports "
             "betting are becoming more and more popular. ")