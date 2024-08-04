# NBA Home Team
[NBA Home Team](https://pp5-ml-dashboard-2d67a903a4d3.herokuapp.com/#results) is a machine learning (ML) project aimed at gaining insight into NBA statistics and building ML pipelines that can predict the outcome of games. We used publicly available data to build two classification pipelines and one clustering pipeline. The two classification pipelines give insight into which statistics may be undervalued when assessing future performance. We also clustered the data to see if the statistics could be used to determine which era of basketball a game belonged to.

## Table of Contents
- [Dataset](#dataset)
- [Business Requirements and Hypotheses](#business-requirements-and-hypotheses)
- [Mapping Business Requirements](#mapping-business-requirements)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Testing](#testing)
- [Deployment](#development)
- [References](#references)


## Dataset
We use a [dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball) from Kaggle. It was collected by Wyatt Walsh (wyattowalsh) using the NBA api. Each row of the table is the stats from a single NBA game. All of the data is public and there are no privacy or ethical issues. 

Out of the 65,698 records, we used 44,909 records. The original dataset has 55 columns. Much of this is metadata. We used 39 of these columns at the beginning of our analysis. Some of our models were trained on fewer for reasons explained in the relevant notebooks. We have taken this subset of the data for two reasons:
- to focus on features relevant to our hypotheses,
- to focus on games with good records.

We will discus the features we focused on in detail in the next section. We restricted our attention to games that:
- occurred during the regular season or the playoffs,
- occurred during or after the 1985-1986 season.

The reason for excluding preseason games is twofold. Many players do not take preseason games seriously since they do not count towards who enters the playoff tournament. Also, they can include exhibition games with teams outside of the NBA.

When looking at the data, the early games have many missing statistics. For example, rebounds, a key statistic in todays understanding of the game, were not kept track of whatsoever. There was also no 3 point line in the earlier days of the NBA. In 1976, there was a merger between the ABA and the NBA. 4 teams were added as well as many players. This naturally impacted the style of play. In 1979, the NBA added the 3 point line, a feature of the ABA. We felt that this marked a reasonable time frame to cut off the data at. However, we chose the 1985 cut off because the data got significantly better. This can be seen in notebook 02. Before 1985, there were approximately 10,000 missing values per season. From 1985 onwards, there are 0 (for the features we are interested in). This was the most compelling argument for us to use this cut off. It also left us with the records of more than 40,000 games, which was sufficient for our analysis.

We removed the games from the 2022-2023 season so that users could use our model to predict the outcome of games that are completely unseen.

### Feature set
Many of the excluded features of the dataset contain metadata, such as whether or not there is video available or what the abbreviation is of the two teams. Some of the data is redundant, such as the percentage based statistics (if we know the number of made and attempted shots, we can sompute the percentage). Here is a list of the remaining features and the meaning of the column name.

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
- `plus_minus_home`: point total of home team minus point total of away team.

The `_away` statistics have the same meaning, but for the opposing team.


## Business Requirements and Hytpotheses

The NBA is the premier basketball league in the world. A fictional online fantasty sports guru has asked us to help give them insight based on the statistics kept by the NBA. This insight will be used to rate players for their fantasy league.

### Business Requirements
- Business Requirement 1: The client has asked us for statistics associated with winning that are less obvious (not related to made shots).

- Business Requirement 2: The client has asked for a model that can predict if the home team wins based on these "less obvious" statistics. The model will then be used to understand the impact/weight of these statistics. The model should have at least an avg precision of 75% and accuracy of 70%.

- Business Requirement 3: The client has asked us for a clustering model in order to determine if meaningful trends can be detected by these methods.

### Hypotheses
- Hypothesis 1: 
  - ML algorithms will gravitate towards the obvious statistics (point related, such as made shots) in order to construct their models.
  - **Validation**: train models on all features and determine which are selected most frequently.

- Hypothesis 2:
  - There are features that don't involve made shots which correlate with winning.
  - **Validation**: correlation study analyzing the relationship between the above features and winning.

- Hypothesis 3:
  - A good pipeline can be built based on features that don't include made shots.
  - **Validation**: drop made shots features and train a model achieving at least 75% avg. precision and 70% accuracy.

- Hypothesis 4:
  - Eras of the NBA can be seen through their statistics.
  - **Validation**: train a clustering model with time removed and use a classification model with time added back to determine the profile of the clusters.



## Mapping Business Requirements
**Business Requirement 1**:

- We need to perform a correlation study to look at what is correlated with wins.
- Pearson's method will detect linear relationships.
- Spearman's method will detect monotonic relationships.
- Predictive Power score will detect more subtle relationships thta may be asymmetric.
- We will also get such statistics from the important features of our finished classification models.
- All, except the last, will be done in the **EDA** epic.

**Business Requirement 2**:

- We need to predict which team won based on the relevant subset of features. 
- We will need to build a binary classification model.
- We will use a standard pipeline to determine the relationship between the features and the target.
- A thorough hyperparameter search will optimize our model for the problem at hand.
- This will be done in the **Classification** epic.

**Business Requirement 3**:
- We need to cluster the data and determine any relationship between the clusters and seasons.
- We will determine the proper number of clusters using the Elbow method and Silhouette scores.
- We will use a standard clustering pipline with PCA to train our initial clustering.
- We will use a classification pipeline to determine the most important features of the clusters and determine an initial profile.
- We will drop the unimportant features, refit the clustering pipeline, 
 and refit the classification pipeline.
- We will use the refit classification to determine a profile by looking at where the features are most distinct among the clusters.
- We will also look at how the season feature is distributed with respect to each cluster.
- This will be done in the **Clustering** epic. 

[TOC](#table-of-contents)

## ML Business Case
### Classification Model
- We want a ML model to predict whether or not the home team wins based on "less obvious statistics" (this means we exclude plus/minus score, total points, and made shots for both home and away teams). the home team winning will be represented as 1 while the home team losing will be represented as 0.
- We will consider classification models (supervised learning) with a two class single label output that matches the target.
- Our goal is a model with:
  - average precision (between win and loss) of 75%
  - accuracy of 70%
- The model will be considered a failure if it fails to achieve these scores. A success will validate Hypothesis 3 and satisfy Business requirement 2.
- The training data to fit the model comes from Kaggle and is described throughly in the [Dataset](#dataset) section of this document.

### Clustering Model
- We want a ML model to cluster the game data in order to see if the statistics reflect eras of basketball.
- We will consider a KMeans clustering model (unsupervised learning).
- We will use a standard clustering pipeline, including principle component analysis.
- We will train our clustering model on the dataset with season removed.
- Our goal is to find the profiles of the clusters and determine their profiles, this will satisfy Business requirement 3.
- To find the profiles of the clusters, we will use a classification model with a multi-class single label output matching targeting the cluster of the data.
- This model will be trained on the same data with season added back in.
- The profiles will be computed by looking at Q1-Q3 of the distributions of each feature with respect to the cluster.
- We will have validated our Hypothesis if in the final profile of the clusters the season range of each cluster have minimal overlap.

[TOC](#table-of-contents)

## Epics and User Stories

The project was split into 6 epics based on the ML tasks. Within each of these Epics, we completed user stories and used the agile methodology.

Epic - Data Collection
* User story (E1US01) - As a **data analyst**, I can import  the dataset from Kaggle so that I can save the data in a local directory.
* User Story (E1US02) - As a **data analyst**, I can load a saved dataset so that I can analyse the data to gain insights on what further tasks may be required.

Epic - Data Visualization, Cleaning, and EDA
* User Story (E2US01) - As a **data scientist**, I can visualise the dataset so that I can interpret which attributes correlate most closely with wins (Business Requirement 1 and Hypothesis 2).
* User Story (E2US02) - As a **data analyst**, I can inspect the dataset to determine what data cleaning tasks should be done.
* User Story (E2US03) - As a **data analyst**, I can test the feature distributions to see if they are normal distributions.
* User Story (E2US04) - As a **data analyst**, I can impute or drop missing data to prepare the dataset.
* User Story (E2US05) - As a **non-technical user**, I can visually inspect the distributions and see relationships between features indicated by correlation coefficients and Predicitve Power score.

Epic - Classification Model: Training, Optimization and Validation
* User Story (E3US01) - As a **data scientist**, I can split the data into a train and test set to prepare it for the ML model.
* User Story (E3US02) - As a **data analyst**, I can determine how to transform the features in order to normalize them.
* User Story (E3US03) - As a **data engineer**, I can fit a ML pipeline with all the data to prepare the ML model for deployment.
* User Story (E3US04) - As a **data scientist**, I can look at the features used by the models in order to determine which are important (Hypothesis 1).
* User Story (E3US05) - As a **data engineer**, I can determine the best algorithm for predicting wins to use in the ML model (Business Requirement 2 and Hypothesis 3).
* User Story (E3US06) - As a **data engineer**, I can carry out an extensive hyperparameter optimisation to ensure the ML model gives the best results (Business Requirement 2 and Hypothesis 3).
* User Story (E3US07) - As a **data scientist**, I can evaluate the model's performance to determine whether it has met our goals for predicting wins (Business Requirement 2 and Hypothesis 3).

Epic - Clustering Model: Training and Evaluation
* User Story (E4US01) - As a **data engineer**, I can determine the number of principal components to use in my pipeline.
* User Story (E4US02) - As a **data engineer**, I can determine the number of clusters to use in my pipeline using the Elbow method and the Silhouette scores.
* User Story (E4US03) - As a **data scientist**, I can use a classification model to predict the clusters games belong to.
* User Story (E4US04) - As a **data engineer**, I can use the classification model to determine the important features for the clustering model.
* User Story (E4US05) - As a **data scientist**, I can use the classification model to produce a profile for the clusters.
* User Story (E4US06) - As a **data scientist**, I can use the profiles to determine if the clusters are related to era (Business Requirement 3 and Hypothesis 4).

Epic - Dashboard Planning, Design, and Development
* User Story (E5US01) - As a **non-technical user**, I can view the project sumamry that describes the project and aspects of it.
* User Story (E5US02) - As a **non-technical user**, I can view the business requirements, hypotheses, and validations to determine how successful the project was.
* User Story (E5US03) - As a **non-technical user**, I can select games the models have not seen and use the models to predict the outcome.
* User Story (E5US04) - As a **technical user**, I can visualize the distributions of the features as well as their correlation, and Predictive Power score (Business Requirement 1 and Hypothesis 2).
* User Story (E5US05) - As a **technical user**, I can view the details of the models and see how they performed on the data (Business Requirement 2  and 3, as well as Hypothesis 3 and 4).
* User Story (E5US06) - As a **non-technical user**, I can examine the profiles of the different clusters and visualize the distributions of the features across ecah cluster.
* User Story (E5US07) - As a **non-technical user**, I can read the conclusions of the project and determine if the hypotheses were validated and if the business requirements were met.

Epic - Deployment
* User Story (E6US01) - As a **user**, I can view the project dashboard on a live website.
* User Story (E6US02) - As a **technical user**, I can learn the details of the project by following along in jupyter notebooks.

[TOC](#table-of-contents)

## Dashboard Design

* Summary Page
  * Introduction
    - Summary of project and motiviation.
  * Dataset
    - Sample of dataset, link to source.
  * Features
    - Meaning of the different statistics.
  * Business Requirements
    - Description of individual business requirements.

* EDA
  * Introduction
    - Different approaches taken during EDA part of project.
  * Feature Distribution
    - Results of Normality testing, plots of distributions colored by target.
  * Correlation and Predictive Power Score
    - Correlation coefficients and scatterplots of pairs of features as well as their Predicitve Power score.
    - Highlight interesting relationships found during EDA.

* Predictor
    - Select games from most recent season in dataset.
    - See the game statistics.
    - Predict outcome and evaluate prediciton with either model.

* Hypotheses and Validation
  * Business Requirements
    - Description of individual business requirements.
  * Hypotheses
    - Statement of each hypothesis along with method of validation.
    - Specifies page on dashboard where user can see the validation.
  * Results
    - Summary of results.

* ML: Naive Feature Selection
  * Hypothesis 1
    - Statment of hypothesis.
  * Process
    - Outline of validation method.
    - Report on findings.
  * Conclusion
    - Summary of results.

* ML: Logistic Regression Model
  * Introduction
    - Restatement of Business Requirement 2.
    - Interesting findings.
  * Hyperparameters
    - Explanation of hyperparameters that were tuned.
    - Outline of search process.
  * Performance Report
    - Evaluation of final model
  * Pipeline
    - Important features and their importance
    - Steps in pipeline

* ML: Adaptive Boost Model
  * Introduction
    - Restatement of Business Requirement 2.
    - Interesting findings.
  * Hyperparameters
    - Explanation of hyperparameters that were tuned.
    - Outline of search process.
  * Performance Report
    - Evaluation of final model
  * Pipeline
    - Important features and their importance
    - Steps in pipeline

* ML: Cluster Analysis
  * Introduction
    - Summary of clustering proceedure.
  * Cluster Profiles
    - Profile dataframe.
    - Smaller profile highlighting season.
    - Description of each cluster profile according to non-overlapping features.
  * Feature Distribution
    - Distribution of seasons for each cluster.
    - Distribution of other features by cluster.
  * Cluster Pipeline
    - Silhouette score diagram.
    - Cluster pipeline steps.
  * Classification Pipeline
    - Training process for classification pipeline.
    - Performance on test set.
    - Important features of classification model.
    - Classification pipeline steps.

* Conclusions
  * Introduction
    - Summary of results.
  * Business Requirements
    - Explanation of how the business requirements were satisfied.
  * Project Outcomes
    - Summary of outcome.
    - Shortcomings.
    - Future directions.

## Deployment
This assumes that you already have a Heroku account.

1. Copy/Clone the repository on github. Check the requirements file and uncomment the relevant packages in order to install the necessary libraries for running the notebooks. Before deploying to Heroku, remember to comment out these packages.
2. Log in to your Heroku account.
3. From the Heroku Dashboard, click the dropdown menu "New" and select "Create new app".
4. Choose a unique name for your app, shoose the appropriate region, and then click "Create app".
5. Add the Config Vars "PORT" with value "8000" and BASE_DIR with no value
6. Scroll down to "Buildpacks". Click "Add buildpack", select "python", and click "Add buildpack".
7. Log in to Heroku from the command line. Execute the command `heroku stack:set heroku-20 --app <your-app-name>`.
8. Check that your requirements file has the appropriate files commented out.
8. Go to the "Deploy" tab. Scroll down to "Deployment method" and select "GitHub". Search for your repository that you copied/cloned in step 1 above. Click "Connect" once you have found it. Scroll down to "Manual deploy" and click "Deploy Branch". Once the build is complete, click "View" to be taken to your deployed app. You may wish to select automatic deployment.

## Testing
### Responsiveness and Accessibility
As the front end of this project is done completely with streamlit, responsiveness was out of oour control as was accessibility, and so we did not address these issues.

### Manual Testing

#### Jupyter Notebooks
Jupyter notebooks were tested by running all cells consecutively. The following User Stories were tested in through the Jupyter notebook:
* E1US01, E1US02
* E2US01-E2US05
* E3US01-E3US07
* E4US01-E4US06
* E6US02

Note, some were also tested in the addressed in the Streamlit app.

#### Streamlit App
Streamlit app was tested manually using user stories. We tested the User Stories from Epics 5 and Epic 6.

Navigation

| Feature                           | Action        | Expected Result | Success |
| --------------------------------- | ------------- | --------------- | ------- |
| Project Summary                   | Click on link | Taken to page   | Yes     |
| Exploratory Data Analysis         | Click on link | Taken to page   | Yes     |
| Predictor                         | Click on link | Taken to page   | Yes     |
| Project Hypotheses and Validation | Click on link | Taken to page   | Yes     |
| ML: Naive Feature Selection       | Click on link | Taken to page   | Yes     |
| ML: Logistic Regression Model     | Click on link | Taken to page   | Yes     |
| ML: Adaptive Boost Model          | Click on link | Taken to page   | Yes     |
| ML: Cluster Analysis              | Click on link | Taken to page   | Yes     |
| Project Conclusions               | Click on link | Taken to page   | Yes     |

Epic - Dashboard Planning, Design, and Development

* User Story (E5US01) - As a **non-technical user**, I can view the project sumamry that describes the project and aspects of it.

| Feature                     | Action                           | Expected Result                                                       | Success |
| --------------------------- | -------------------------------- | --------------------------------------------------------------------- | ------- |
| Project Summary page        |                                  |                                                                       |         |
| Description of methods      | View summary page                | Get descripting of methods                                            | Yes     |
| View Dataset                | Click Dataset link               | Taken to Dataset section and see sample data                          | Yes     |
| Dataset Feature description | Click Features link              | Taken to Features section and read feature descriptions               | Yes     |
| Business Requirements       | Click Business Requirements link | Taken to Business Requirements section and read business requirements | Yes     |

* User Story (E5US02) - As a **non-technical user**, I can view the business requirements, hypotheses, and validations to determine how successful the project was.

| Feature                                | Action                           | Expected Result                                                       | Success |
| -------------------------------------- | -------------------------------- | --------------------------------------------------------------------- | ------- |
| Project Hypotheses and Validation page |                                  |                                                                       |         |
| Business Requirements                  | Click Business Requirements link | Taken to Business Requirements section and read business requirements | Yes     |
| Hypotheses and Validation              | Click Hypotheses link            | Taken to Hypotheses link and read hypotheses and validation methods   | Yes     |
| View Results Summary                   | Click Results link               | Taken to Results section and read summary of results                  | Yes     |

* User Story (E5US03) - As a **non-technical user**, I can select games the models have not seen and use the models to predict the outcome.

| Feature                        | Action                                       | Expected Result                     | Success |
| ------------------------------ | -------------------------------------------- | ----------------------------------- | ------- |
| Predictor page                 |                                              |                                     |         |
| Select Home Team               | Select team from dropdown menu               | Possible games are updated          | Yes     |
| Select Away Team               | Select team from dropdown menu               | Possible games are updated          | Yes     |
| Select Game                    | Select game from dropdown menu               | Statistics are updated              | Yes     |
| Logistic Regression Prediction | Click Predict with Logistic Regression       | Prediction displayed,and evalutated | Yes     |
| Adaptive Boost Prediction      | Click Predict with Adaptive Boost Classifier | Prediction displayed,and evalutated | Yes     |

* User Story (E5US04) - As a **technical user**, I can visualize the distributions of the features as well as their correlation, and Predictive Power score (Business Requirement 1 and Hypothesis 2).

| Feature                          | Action                                            | Expected Result                                                                          | Success |
| -------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------- |
| EDA page                         |                                                   |                                                                                          |         |
| Summary of EDA                   | Read summary                                      | Learn what was done during EDA phase                                                     | Yes     |
| Feature Distribution section     | Click Feateure Distribution link                  | Taken to Feature Distribution section                                                    | Yes     |
| Feature Normality test score     | Select feature from dropdown menu                 | See score of feature on normality test and associated p-value                            |         |
| Feature distribution plot        | Select feature from dropdown menu                 | See histogram plotting feature for home and away teams and colored by value of Home Wins | Yes     |
| Correlation and PPS section      | Click Correlation and Predictive Power score link | Taken to Correlation and Pedictive Power score section                                   | Yes     |
| Correlation and PPS for features | Select two features from two dropdown menus       | See scatterplot relating features, their correlation coefficient and pps                 | Yes     |
| Interesting relationship         | Read interesting relationships section            | Learn interesting relationships found through EDA                                        | Yes     |

* User Story (E3US04) - As a **data scientist**, I can look at the features used by the models in order to determine which are important (Hypothesis 1).

| Feature                             | Action                    | Expected Result                                                         | Success |
| ----------------------------------- | ------------------------- | ----------------------------------------------------------------------- | ------- |
| ML: Naive Feature Selection page    |                           |                                                                         |         |
| Statement of Hypothesis 1           | Click on Hypthesis 1 link | Taken to Hypothesis 1 section and learn about Hypothesis 1              | Yes     |
| Process for validation              | Click on Process link     | Taken to Process section and learn validation method for Hypothesis 1   | Yes     |
| Conclusions of testing Hypothesis 1 | Click on Conclusions link | Taken to Conclusions section and learn results of our validation method | Yes     |

* User Story (E5US05) - As a **technical user**, I can view the details of the models and see how they performed on the data (Business Requirement 2  and 3, as well as Hypothesis 3 and 4).

| Feature                                       | Action                               | Expected Result                                                                                            | Success |
| --------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ------- |
| ML: Logistic Regression Model page            |                                      |                                                                                                            |         |
| Summary of Logistic Regression model training | Read summary                         | Learn about steps in producing Logistic Regression model                                                   | Yes     |
| Hyperparameter search                         | Click on the Hyperparameters link    | Learn the value of the hyperparameters used in the final model and about the hyperparameter search process | Yes     |
| Performance Report                            | Click on the Performance Report link | Learn how the model performed and evaluate the model                                                       | Yes     |
| Pipeline                                      | Click on the Pipeline link           | Learn about the important features to the model and the steps in the pipeline                              | Yes     |
| ML: Adaptive Boost Model page                 |                                      |                                                                                                            |         |
| Summary of Logistic Regression model training | Read summary                         | Learn about steps in producing Adaptive Boost model                                                        | Yes     |
| Hyperparameter search                         | Click on the Hyperparameters link    | Learn the value of the hyperparameters used in the final model and about the hyperparameter search process | Yes     |
| Performance Report                            | Click on the Performance Report link | Learn how the model performed and evaluate the model                                                       | Yes     |
| Pipeline                                      | Click on the Pipeline link           | Learn about the important features to the model and the steps in the pipeline                              | Yes     |

* User Story (E5US06) - As a **non-technical user**, I can examine the profiles of the different clusters and visualize the distributions of the features across ecah cluster.

| Feature                               | Action                                                        | Expected Result                                                                       | Success |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------- |
| ML: Cluster Analysis page             |                                                               |                                                                                       |         |
| Summary of Cluster Analysis           | Read summary                                                  | Learn about the steps taken in the cluster analysis                                   | Yes     |
| Cluster Profiles                      | Click Cluster Profiles link                                   | Taken to Cluster Profiles section and learn the profiles of the clusters              | Yes     |
| Feature Distribution section          | Click Feature Distribution link                               | Taken to Feature Distribution section                                                 | Yes     |
| Distribution of seasons over clusters | Look at tables displaying distribution of seasons per cluster | Learn how the seasons are distrubed over each cluster                                 | Yes     |
| Distribution of other features        | Select a feature from dropdown menu                           | Learn how the feature is distributed over each cluster                                | Yes     |
| Cluster Pipeline                      | Click Cluster Pipeline link                                   | See the Silhoutte scores for the pipeline and the steps in the pipeline               | Yes     |
| Classification Pipeline               | Click the Classification Pipeline link                        | See the performance report for the classification model and the steps in the pipeline | Yes     |

* User Story (E5US07) - As a **non-technical user**, I can read the conclusions of the project and determine if the hypotheses were validated and if the business requirements were met.

| Feature                  | Action                           | Expected Result                                                                                  | Success |
| ------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------ | ------- |
| Project Conclusions page |                                  |                                                                                                  |         |
| Summary of conclusions   | Read summary of conclusions      | Learn if the project was a success or not                                                        | Yes     |
| Business Requirements    | Click Business Requirements link | Taken to Business Requirements section and learn how they were satisfied                         | Yes     |
| Project Outcomes         | Click Project Outcomes link      | Taken to Project Outcomes section, read evaluation of whole project, and about future directions | Yes     |

* User Story (E6US01) - As a **user**, I can view the project dashboard on a live website.
  * The existence of the live site tests this User Story.



### Validation
To validate my python code I used flake8. In the end, the code had the following remaining issues:

- It claimed the env file was imported but not used. This is incorrect, but flake8 can not tell.
- flake8 complained about arctictern.py and make_url.py which are not files that I have edited.

I did not validate any other code as the app is built with streamlit and so it generates all relevant HTML and JavaScript. 

### Bugs

#### Bugs Fixed
A lot of debugging took place during the writing of the notebooks and was therefore not documented.

* Bug: Streamlit not loading anything other than basic text and select boxes.
  * Fix: The plot was not loading because I wasn't using the proper streamlit commands.

* Bug: qq-plots stopped working.
  * Fix: This was due to a change in the versions of some of the packages. In order to fix this, I used SciPy instead to generate the same plots instead of Pingouin.

* Bug: Jarque-Bera test for normality no longer working.
  * Fix: This was due to a change in the versions of some of the packages. I used SciePy to do the test and generated custom code to produce a result similar to what Pingouin was producing.

* Bug: The find_feature is producing kept and drop lists that overlap.
  * Fix: Something was being added to one of the lists in the incorrect conditional statement.

* Bug: When visualizing the Elbow method or the Silhouette scores with yellowbrick, there is a missing font warning/notification.
  * Fix: Changing the versions of certain packages addressed this.

* Bug: When trying to do a grid search, I got the error Nonetype object has no attribute set_params.
  * Fix: I was using the default base estimator and didn't realize that if I wanted to modify the parameters of the internal base estimator I needed to still pass the base estimetor to the Adaptive Boost model. Note, this is no longer relevant since I did not work with parameters of the base estimator for the Adaptive Boost model.

* Bug: Grid search produced hundreds of warnings such as FutureWarning and DataConversionWarning.
  * Fix: I found multiple solutions but none of them seemed to work on their own. In the end, I combined suppressing the warnings using the warning package, using logging to capture the warnings, and setting an environment variable.

* Bug: During the generation of Profile Reports by ydata package, I got the error "Error rendering output item using 'jupyter-ipywidget-renderer'".
  * Fix: This error was addressed by running the notebooks in the browser with gitpod instead of on my laptop remotely using gitpod.

* Bug: Some tables on the EDA page were not displaying.
  * Fix: The function I was using was not taking care of the edge cases `home_wins` and `plus_minus_home`.

#### Bugs Left In

* Bug: Prediction button sometimes redirects to Summary page.
  * Reason for leaving in: No error or warning message is given and the bug is not consistently reproducible. Navigating back to the Predictor page and trying again yields a successful prediction.

* Bug: Some notebooks, like 07, become quite slow.
  * Reason for leaving in: I can not figure out the cause of this. temporary fix seems to be making a copy of the notebook file and working with the copy. I tried looking online, contacting Tutor Support, and the Predictive Analytics slack channel. Tutor Support was unable to help and the request on the predictive analytics channel did not generate a single reply.

* Bug: The data set is missing regular season data from the 2012-2013 season.
  * Reason for leaving in: This is not entirely a bug. It was noticed only during notebook 07. I did not retrieve the missing data and add it to the dataset because working with the NBA api is out of the scope of the project and I didn't feel it significantly impacted the results of the project.





## References
### Influences

* I want to thank Mo, my excellent mentor. He was very helpful in multiple ways. He helped me focus on attainable goals, addressed questions I had, and provided me with instructive examples.

* [This project](https://github.com/jfpaliga/CVD-predictor?tab=readme-ov-file#technologies-used) was an inspirational guide. It helped with understanding the CRISP-DM framework as well as how to structure user stories for such a project. I did use it as an example of how to structure certain things.

* I learned a lot doing the Churnometer walkthrough project. I borrowed some functions directly for the analysis of transformations and important features, this is mentioned in the notebooks and code. The methodology for many of the pipelines that I built was taken from this project.

* I gained a lot from talking with my fellow students Tarek and Tariq.

* I spent time talking to my friend Louis Casinelli about basketball. He gave me helpful insight as he is more knowledgable on the topic.


### Technologies Used
Python was the main technology used as well as various Machine Learning libraries. I made extensive use of Jupyter Notebooks to document and construct the project. I also used Streamlit to build a web app and Heroku to host said app.

- Python Packages: streamlit, altair, pandas, matplotlib, seaborn, ydata-profiling, feature-engine, scikit-learn, protobuf, yellowbrick, Jinja2, MarkupSafe, pingouin, ppscore, ipywidgets, ipython, xgboost, and numpy

- flake8 for python validation.

- Documentation for pandas, scikit-learn, pingouin, scipy, and streamlit.

- Google Docs spreadsheets and [Table to Markdown](https://tabletomarkdown.com/convert-spreadsheet-to-markdown/) to make the markdown tables in the testing section.

- Github and Git for verison control.

- VScode and Gitpod as development environments.

- Codeium and Sorcery as advanced auto-complete tools.

- Black to format python.

### Specific Links
#### StackOverflow
- [Duplicating git repo](https://stackoverflow.com/questions/6613166/how-to-duplicate-a-git-repository-without-forking)
- [Suppress warnings 1](https://stackoverflow.com/questions/52224813/python-warnings-filterwarnings-does-not-ignore-deprecationwarning-from-import-s) and [Suppress warnings 2](https://stackoverflow.com/questions/879173/how-to-ignore-deprecation-warnings-in-python)
- [Finding importance of features for Logistic Regression model](https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model)
- [All estimators in grid search](https://stackoverflow.com/questions/65359261/can-you-get-all-estimators-from-an-sklearn-grid-search-gridsearchcv)
- [Iterating over dataframe rows](https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe)
- [Checking for nan values](https://stackoverflow.com/questions/944700/how-to-check-for-nan-values)
- [String dict to dict](https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary)
- [Modify pipeline parameters](https://stackoverflow.com/questions/63963914/change-a-parameter-of-a-sklearn-pipeline)
- [Yellowbrick import error](https://stackoverflow.com/questions/65602076/yellowbrick-importerror-cannot-import-name-safe-indexing-from-sklearn-utils)
- [Rotate tick labels in seaborn](https://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot)

#### Other
- [Kaggle: Wyatt Walsh's Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball)
- [Reddit: Importance of preseason](https://www.reddit.com/r/nba/comments/173gab7/does_nba_preseason_actually_matter/)
- [Wikipedia: ABA and NBA merger](https://en.wikipedia.org/wiki/ABA%E2%80%93NBA_merger#ABA_contributions_to_NBA_play)
- [Conditionally delete pandas rows](https://saturncloud.io/blog/python-pandas-conditionally-delete-rows/#:~:text=We%20can%20aslo%20use%20the,met%2C%20similar%20to%20boolean%20indexing)
- [Normality and Jarque Bera Test](https://groups.google.com/g/pystatsmodels/c/ILPrX08Fl08)
- [Tuning Adaptive Boost models](https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20)
- [Predictive Power score](https://macrosynergy.com/research/the-predictive-power-score/)
- [Hyperparameters for classification models](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)
- [Approaching ML problems](https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur/)
- [Plot multiple graphs](https://learnt.io/blog/how-to-plot-multiple-graphs-in-python/)
