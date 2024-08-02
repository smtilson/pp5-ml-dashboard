# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
from src.utils import get_df
import joblib
from src.inspection_tools import (get_matchups, get_dates, lookup_game,
                                  prepare_game_data)

# from src.data_management import load_telco_data, load_pkl_file


def page_predictor_body():

    # Introduction
    st.write("# Predictor")
    st.write(
        "Using our two models, you can predict the outcome of a game "
        "from the 2022-2023 season, which our models were not trained or "
        "tested on."
    )
    # Load Data
    latest_season = get_df("latest_season", "datasets/clean/csv")
    version = "v1"
    file_path = f"outputs/ml_pipeline/predict_home_wins/{version}"
    logistic_path = file_path + "/logistic_pipeline.pkl"
    logistic_pipe = joblib.load(filename=logistic_path)
    ada_path = file_path + "/ada_pipeline.pkl"
    ada_pipe = joblib.load(filename=ada_path)

    # Selection
    home_team, away_team = draw_team_selection_widget(latest_season)
    game_id, game_date = draw_game_selection_widget(latest_season, home_team,
                                                    away_team)
    game_data = prepare_game_data(latest_season, game_id)
    game_data = game_data.astype("int32")
    st.write(f"## {home_team} vs. {away_team} on {game_date}")
    st.write("### Game Stats")
    st.dataframe(game_data)

    # Prediction
    st.write("## Prediction")
    st.write(
        "Our models see a subset of the above data. See the page "
        "related to the model in question for more details."
    )
    model_name = "Classifier"
    prediction = -1
    prob = None
    if st.button("Predict with Logistic Regression"):
        prediction, prob, outcome = make_prediction(
            logistic_pipe, latest_season, game_id
        )
        model_name = "Logistic Regression"
    if st.button("Predict with Adaptive Boost Classifier"):
        prediction, prob, outcome = make_prediction(ada_pipe, latest_season,
                                                    game_id)
        model_name = "Adaptive Boost Classifier"
    if prediction >= 0:
        interpret(model_name, home_team, away_team, prediction, prob, outcome)


def get_teams(df):
    teams = list(df["team_name_home"].unique())
    teams += list(df["team_name_away"].unique())
    return list(set(teams))


def draw_team_selection_widget(df):
    teams = get_teams(df)
    home_index = teams.index("Denver Nuggets")
    away_index = teams.index("Minnesota Timberwolves")
    home_team = st.selectbox(label="Home Team", options=teams,
                             index=home_index)
    away_team = st.selectbox(label="Away Team", options=teams,
                             index=away_index)
    return home_team, away_team


def draw_game_selection_widget(df, home_team, away_team):
    matchups = get_matchups(df, home_team, away_team)
    dates = get_dates(matchups)
    game_date = st.selectbox(label="Date", options=dates, index=0)
    return lookup_game(df, home_team, away_team, game_date), game_date


def make_prediction(pipe, df, game_id):
    row = df.loc[game_id]
    game = pd.Series.to_frame(row).T
    outcome = game["home_wins"].iloc[0]
    drop_list = [
        "team_name_home",
        "team_name_away",
        "day",
        "month",
        "year",
        "home_wins",
    ]
    game.drop(drop_list, axis=1, inplace=True)
    game = game.astype("int32")
    return pipe.predict(game), pipe.predict_proba(game), outcome


def interpret(model_name, home_team, away_team, prediction, prob, outcome):
    prob = prob[0]
    msg = f"Our {model_name} model has predicted that "
    if prediction == 1:
        msg += f"the {home_team} will beat the {away_team} with a probability "
        msg += f"of **{prob[1]*100:.2f}%**."
    elif prediction == 0:
        msg += f"the {away_team} will beat the {home_team} with a probability "
        msg += f"of **{prob[0]*100:.2f}%**."
    if prediction == outcome:
        msg += "\n\nOur model is **correct**!"
    else:
        msg += "\n\nOur model is **incorrect**."
    st.write(msg)
