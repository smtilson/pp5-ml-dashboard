# NBA Home Team Wins

## Dataset
We use a dataset from Kaggle. It was collected by Wyatt Walsh (username) using the NBA api. Each row of the table is the stats from a single NBA game. All of the data is public and there are no privacy or ethical issues. 

Out of the 65,698 records, we used 44,909 records. The original dataset has 55 columns. Much of this is metadata. We used 39 of these columns at the beginning of our analysis. Some of our models were trained on fewer for reasons explained in the relevant notebooks. We have taken this subset of the data for two reasons:
- to focus on features relevant to our hypotheses,
- to focus on games with good records.

We will discus the features we focused on in detail in the next section. We restricted our attention to games that:
- occurred during the regular season or the playoffs,
- occurred during or after the 1985-1986 season.

The reason for excluding preseason games is twofold. Many athletes do not take preseason games seriously since they do not count towards who enters the playoff tournament. Also, they can include exhibition games with teams outside of the NBA.

When looking at the data, much of the records of the early game is missing. Many key statistics, by todays understanding of the game, were not kept track of whatsoever (such as rebounds). There was also no 3 point line in the earlier days of the NBA. In 1976, there was a merger between the ABA and the NBA. 4 teams were added as well as many players. The style of play changed as well. In 1979, the NBA added the 3 point line, a feature of the ABA. We felt that this marked a reasonable time frame to cut off the data at. However, we chose the 1985 cut off because the data got significantly better. This can be seen in notebook 02. Before 1985, there were approximately 10,000 missing values per season. From 1985 onwards, there are 0 (for the features we are interested in). This was the most compelling argument for us to use this cut off. It also left us with the records of more than 40,000 games, which was sufficient for our analysis.

### Feature set
Many of the excluded features of the dataset contain metadata, such as whether or not there is video available or what the abbreviation is of the two teams. Some of the data is redundant, such as the percentage based statistics. As we know the number of attempted shots and the number of made shots, the percentage of made shots is a redundant feature. Here is a list of the remaining features and the meaning of the column name.

- `game_id`: a unique identifier of each game record, also contains data about which season it is a member of.
- `season_id`: identifies which season and which type of game (playoffs or regular season) the record is for.
- `team_id_home`: a unique identifier for the team/franchise as team names and cities can change over time.
- `wl_home`: W if the home team won, L if the home team lost.
- `fgm_home`: number of field goals (2 point and 3 point shots) made by the home team. Does not include free throws (foul shots).
- `fga_home`: number of field goals attempted by the home team.
- `fg3m_home`: number of 3 point field goals (shots) made by the home team.
- `fg3a_home`: number of 3 point shots attempted by the home team.
- `ftm_home`: number of free throws (foul shots, they are worth only 1 point) made by the home team.
- `fta_home`: number of free throws attempted by the home team.
- `oreb_home`: number of offensive rebounds made by the home team.
- `dreb_home`: number of defensive rebounds made by the home team.
- `reb_home`: number of total rebounds made by the home team.
- `ast_home`: number of assists by the home team.
- `stl_home`: number of steals made by the home team.
- `blk_home`: number of shots blocked by the home team.
- `tov_home`: number of times the home team turned over the ball.
- `pf_home`: number of personal fouls the home team commited.
- `pts_home`: total points scored by the home team.
- `plus_minus_home`: point differential, ',
       'team_id_away', 'fgm_away', 'fga_away', 'fg3m_away', 'fg3a_away',
       'ftm_away', 'fta_away', 'oreb_away', 'dreb_away', 'reb_away',
       'ast_away', 'stl_away', 'blk_away', 'tov_away', 'pf_away', 'pts_away',
       'Day', 'Month', 'year'