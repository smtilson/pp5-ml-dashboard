# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.utils import get_df
import joblib
from src.inspection_tools import get_matchups, get_dates, lookup_game, prepare_game_data
#from src.data_management import load_telco_data, load_pkl_file


def page_predictor_body():
    st.write("## Predictor")
    st.write("Using our two models, you can predict the outcome of a game "\
             "from the 2022-2023 season, which our models have not yet seen.")
    latest_season = get_df("latest_season","datasets/clean/csv")
    version = 'v1'
    logistic_path = f"workspace/"
    logistic_pipe = joblib.load(filename=file_path) 
    ada_pipe = 1
    
    
    game_id = draw_selection_widget(latest_season)
    game_data = prepare_game_data(latest_season, game_id)
    game_data = game_data.astype("int32")
    home_team = game_data.iloc[0].index
    awa_team = game_data.iloc[1].index
    st.write(f"### {home_team} vs. {away_team} on {game_date}")
    st.write("### Game Stats")
    st.table(game_data)
    
    
    
    

def get_teams(df):
    teams = list(df['team_name_home'].unique())
    teams += list(df['team_name_away'].unique())
    return list(set(teams))

def draw_selection_widget(df):
    teams = get_teams(df)
    home_index = teams.index('Denver Nuggets')
    away_index= teams.index('Minnesota Timberwolves')
    home_team = st.selectbox(label="Home Team", options=teams, index=home_index)
    away_team = st.selectbox(label="Away Team", options=teams, index=away_index)
    matchups = get_matchups(df, home_team, away_team)
    dates = get_dates(matchups)
    game_date = st.selectbox(label="Game Date (DD/MM/YYYY)",
                                        options=dates, index=0)
    game_id = lookup_game(df, home_team, away_team, game_date)
    return game_id
def make_prediction(pipe, df, game_id):
    pass