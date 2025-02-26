# This file follows that of the Churnometer walkthrough project
import streamlit as st
from src.utils import get_df


def page_summary_body():
    # TOC
    st.write("* [Dataset](#dataset)")
    st.write("* [Features](#features)")
    st.write("* [Business Requirements](#business-requirements)")

    # Introduction
    st.write("# Summary")
    st.write(
        "Fantasy sports is a huge industry. In 2022, the size of the "
        "market was [estimated](https://www.skyquestt.com/report/fantasy"
        "-sports-market) at $27 billion. We were approached by a "
        "fictional sports analyst with a focus on fantasy basketball. "
        "They asked us to study NBA statistics to determine patterns as "
        "well as new statistics that could be used to predict the "
        "winning team."
    )
    st.write(
        "We used various techniques from machine learning to address the"
        " business concerns. We performed exploratory data analysis, "
        "trained classification models, and trained a clustering model. "
        "Please see [NBA Home Team Wins](https://github.com/smtilson/pp5-ml-"
        "dashboard) for more details about the project."
    )

    # load data
    game_data = get_df("game_pre_split", "datasets/clean/csv")

    # Content
    st.write("## Dataset")
    st.info(
        "We used a dataset from Kaggle generated by Wyatt Walsh using the"
        " NBA api. Here is a sampling."
    )
    st.dataframe(game_data.head(10))
    st.info(
        "* **Dataset**: The data is all publicly available [here]"
        "(https://www.kaggle.com/datasets/wyattowalsh/basketball).\n"
        "* **Dataset Profile**: After preparation, there are 44,909 games "
        f"recorded with {len(game_data.columns)} columns. Home Wins is the "
        "target."
    )

    st.write("## Features")
    st.write(
        "- Game ID: a unique identifier of each game record.\n"
        "- Season: how many seasons after the 1985-1986 season the game "
        "occured during..\n"
        "- Home Wins: 1 if the home team won, 0 if the away team won.\n"
        "- FGM Home: number of field goals (2 point and 3 point shots) "
        "made by the home team. Does not include free throws (foul shots)"
        ".\n"
        "- FGA Home: number of field goals attempted by the home team."
        "\n"
        "- FG3M home: number of 3 point field goals (shots) made by the"
        " home team.\n"
        "- FG3A Home: number of 3 point shots attempted by the home "
        "team.\n"
        "- FTM Home: number of free throws (foul shots, they are worth "
        "only 1 point) made by the home team.\n"
        "- FTA Home: number of free throws attempted by the home team."
        "\n"
        "- OREB Home: number of offensive rebounds made by the home "
        "team.\n"
        "- DREB Home: number of defensive rebounds made by the home "
        "team.\n"
        "- REB Home: number of total rebounds made by the home team.\n"
        "- AST Home: number of assists by the home team.\n"
        "- STL Home: number of steals made by the home team.\n"
        "- BLK Home: number of shots blocked by the home team.\n"
        "- TOV Home: number of times the home team turned over the "
        "ball.\n"
        "- PF Home: number of personal fouls the home team commited.\n"
        "- PTS Home: total points scored by the home team.\n"
        "- Plus Minus Home: point total of home team minus point total "
        "of away team.")
    st.write("The `_away` statistics have the same meaning, but for the "
             "opposing team.")

    st.write("## Business Requirements")
    st.info(
        "- **Business Requirement 1**: The client has asked us for "
        "statistics associated with winning that are less obvious (points "
        "and made shots).\n"
        "- **Business Requirement 2**: The client has asked for a model "
        'that can predict if the home team wins based on these "less '
        'obvious" statistics. The model will then be used to understand the'
        " impact/weight of these statistics. The model should have at "
        "least an avg precision of 75% and accuracy of 70%.\n"
        "- **Business Requirement 3**: The client has asked us for a "
        "clustering model in order to determine if meaningful trends can be"
        " detected by these methods."
    )
