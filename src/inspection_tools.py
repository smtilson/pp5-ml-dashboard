# These functions are explicitly for inspecting the various dataframes we encounter.
import pandas as pd

def single_season(df,target_season_id) -> 'DataFrame':
    return df.query(f'season_id == {target_season_id}')

def compute_years(df) -> list:
    return df['Year'].unique()

def cutoff_year(season_id:int,cutoff:int) -> bool:
    year_str =str(season_id)[1:]
    if len(year_str)>4:
        raise ValueError("season_id is too long.")
    year = int(year_str)
    return year >= cutoff


# This was gotten from the StackOverflow answer at
# https://stackoverflow.com/questions/70748529/how-to-save-pandas-info-function-output-to-variable-or-data-frame
def get_info_df(df:'DataFrame') -> 'DataFrame':
    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().splitlines()
    df = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
       .drop('Count',axis=1)
       .rename(columns={'Non-Null':'Non-Null Count'}))
    return df

def info_dtype_dict(df:'DataFrame') -> dict:
    info_df = get_info_df(df)
    data_types = {row['Column']:row['Dtype'] for index,row in info_df.iterrows()}
    return data_types

def get_season_df(df,beginning_year) -> 'DataFrame':
    return df.query(f'season_id in []')

def season_data(game_df) -> 'DataFrame':
    season_ids = game_df['season_id'].unique()
    years_dict = {id:compute_years(single_season(game_df,id)) for id in season_ids}
    season_type_dict = {id:game_df.query(f'season_id=={id}')['season_type'].unique() for id in season_ids}
    game_df['years'] = game_df.apply(lambda x: years_dict[x['season_id']], axis=1)
    season_data = pd.DataFrame()
    season_data['season_id'] = game_df['season_id'] 
    season_data.drop_duplicates(inplace=True)
    season_data['years'] = game_df['years']
    season_data['game_types'] = season_data.apply(lambda x: season_type_dict[x['season_id']], axis=1)
    return season_data
