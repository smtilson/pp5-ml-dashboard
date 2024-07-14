from utils import get_df, get_info_df, info_dtype_dict

import pandas as pd
import numpy as np

games = get_df('game')
line_scores = get_df('line_score')
other_stats = get_df('other_stats')
team_info_common = get_df('team_info_common')