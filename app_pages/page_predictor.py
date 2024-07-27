# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.utils import get_df
#from src.data_management import load_telco_data, load_pkl_file


def page_predictor_body():
    st.write("## Predictor")
    st.write("In progress...")
    st.write("Using our two models, you can predict the outcome of a game "\
             "from the 2022-2023 season, which was not included in our "\
             "training or test data.")
    latest_season = get_df("latest_season","datasets/clean/csv")
    teams = get_teams(latest_season)
    with st.form("Game Selection"):
        home_team = st.selectbox(label="Home Team", options=teams, index=0)
        away_team = st.selectbox(label="Away Team", options=teams, index=1)
        submitted = st.form_submit_button("Get Game List")
        if submitted:
            games = latest_season.query(f'team_name_home== "{home_team}" & team_name_away == "{away_team}"')
            game_dates = []
            for row in games.iterrows():
                date = row['day']+"/"+row['month']+"/"+row['year']
                game_dates.append(date)
            game_dates = list(set(game_dates))
            game_date = st.selectbox(label="Game Date", options=game_dates, index=0)
    #date = st.selectbox(options=game_dates, index=0)
    #full_data = get_df()

def get_teams(df):
    teams = list(df['team_name_home'].unique())
    teams += list(df['team_name_away'].unique())
    return list(set(teams))

def find_game(df,home_team,away_team,date):
    pass