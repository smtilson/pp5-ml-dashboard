# These functions are explicitly for inspecting the various dataframes we
# encounter.
import pandas as pd
import io
from src.utils import disp


def single_season(df, target_season_id) -> pd.DataFrame:
    return df.query(f"season_id == {target_season_id}")


def compute_years(df) -> list:
    return df["year"].unique()


def cutoff_year(season_id: int, cutoff: int) -> bool:
    year_str = str(season_id)[1:]
    if len(year_str) > 4:
        raise ValueError("season_id is too long.")
    year = int(year_str)
    return year >= cutoff


# This was gotten from the StackOverflow answer at
# https://stackoverflow.com/questions/70748529/how-to-save-pandas-info-function-output-to-variable-or-data-frame
def get_info_df(df: pd.DataFrame) -> pd.DataFrame:
    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().splitlines()
    df = (
        pd.DataFrame([x.split() for x in lines[5:-2]],
                     columns=lines[3].split())
        .drop("Count", axis=1)
        .rename(columns={"Non-Null": "Non-Null Count"})
    )
    return df


def info_dtype_dict(df: pd.DataFrame) -> dict:
    info_df = get_info_df(df)
    return {row["Column"]: row["Dtype"] for index, row in info_df.iterrows()}


def get_season_df(df, beginning_year) -> pd.DataFrame:
    return  # df.query(f'season_id in []')


def season_data(game_df) -> pd.DataFrame:
    season_ids = game_df["season_id"].unique()
    years_dict = {id: compute_years(single_season(game_df, id))
                  for id in season_ids}
    season_type_dict = {
        id: game_df.query(f"season_id=={id}")["season_type"].unique()
        for id in season_ids
    }
    game_df["years"] = game_df.apply(lambda x: years_dict[x["season_id"]],
                                     axis=1)
    season_data = pd.DataFrame()
    season_data["season_id"] = game_df["season_id"]
    season_data.drop_duplicates(inplace=True)
    season_data["years"] = game_df["years"]
    season_data["game_types"] = season_data.apply(
        lambda x: season_type_dict[x["season_id"]], axis=1
    )
    return season_data


def reduce_corr_df(df_corr, threshold):
    for col in df_corr.columns:
        try:
            maxi = df_corr[col].nlargest(2)[1]
        except Exception as e:
            print(df_corr[col].nlargest(2))
            raise e
        mini = df_corr[col].min()
        # print(col,mini,maxi,threshold)
        if max(abs(maxi), abs(mini)) < threshold:
            # print('removing', col)
            df_corr.drop(col, axis=1, inplace=True)
    for row in df_corr.index:
        try:
            maxi = df_corr.loc[row].nlargest(2)[1]
        except Exception as e:
            print(df_corr.loc[row].nlargest(2))
            raise e
        mini = df_corr.loc[row].min()
        # print(row,mini,maxi,threshold)
        if max(abs(maxi), abs(mini)) < threshold:
            # print('removing', row)
            df_corr.drop(row, axis=0, inplace=True)
    return df_corr


def get_matchups(df, team_1, team_2):
    matchups = df.query(
        f'team_name_home == "{team_1}" & ' f'team_name_away == "{team_2}"'
    )
    relevant_cols = [
        "season_id",
        "day",
        "month",
        "year",
        "team_name_home",
        "team_name_away",
        "home_wins",
        "pts_home",
        "pts_away",
    ]
    return matchups.filter(relevant_cols)


def get_date(row):
    day, month, year = int(row["day"]), int(row["month"]), int(row["year"])
    day, month, year = str(day), str(month), str(year)
    return f"{day}/{month}/{year}"


def get_dates(df):
    return [get_date(row) for _, row in df.iterrows()]


def lookup_game(df, home_team, away_team, date):
    matchups = get_matchups(df, home_team, away_team)
    for index, row in matchups.iterrows():
        if get_date(row) == date:
            return index


def prepare_game_data(df, index):
    row = df.loc[index]
    home_team_data = {}
    away_team_data = {}
    for col in df.columns:
        if "_home" in col:
            if "plus_minus" in col:
                continue
            new_col = disp(col.replace("_home", ""))
            home_team_data[new_col] = row[col]
        elif "_away" in col:
            new_col = disp(col.replace("_away", ""))
            away_team_data[new_col] = row[col]
    data_dict = {key: [val] for key, val in home_team_data.items()}
    for key, val in away_team_data.items():
        data_dict[key].append(val)
    df = pd.DataFrame.from_dict(data_dict)
    df.set_index("TEAM NAME", inplace=True)
    df.astype(str)
    return df
