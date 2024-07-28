# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.utils import get_df
from src.inspection_tools import get_matchups, get_dates, lookup_game
#from src.data_management import load_telco_data, load_pkl_file


def page_predictor_body():
    st.write("## Predictor")
    st.write("In progress...")
    st.write("Using our two models, you can predict the outcome of a game "\
             "from the 2022-2023 season, which was not included in our "\
             "training or test data.")
    latest_season = get_df("latest_season","datasets/clean/csv")
    teams = get_teams(latest_season)
    
    home_index = teams.index('Denver Nuggets')
    away_index= teams.index('Minnesota Timberwolves')
    with st.form("Game Selection"):
        home_team = st.selectbox(label="Home Team", options=teams, index=home_index)
        away_team = st.selectbox(label="Away Team", options=teams, index=away_index)
        if st.form_submit_button("Get List of Matchups"):
            matchups = get_matchups(latest_season, home_team, away_team)
            dates = get_dates(matchups)
            game_date = st.selectbox(label="Game Date (DD/MM/YYYY)", options=dates, index=0)
    st.write(f"## {home_team} vs. {away_team} on {game_date}")
    st.write(lookup_game(latest_season, home_team, away_team, game_date))
    #date = st.selectbox(options=game_dates, index=0)
    #full_data = get_df()

def get_teams(df):
    teams = list(df['team_name_home'].unique())
    teams += list(df['team_name_away'].unique())
    return list(set(teams))

def find_game(df, home_team, away_team,date):
    matchups = df.query(f'team_name_home == "{home_team}" &'\
                        f'team_name_away == "{away_team}"')
    day, month, year = date.split("/")
    game = matchups.query(f'day == "{day}" & month == "{month}" &'\
                          f'year == "{year}"')
    return game.index