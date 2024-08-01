# This file follows that of the Churnometer walkthrough project
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from src.utils import get_df, BASE_DIR
from src.display import display_report

# from src.data_management import load_telco_data, load_pkl_file


def page_cluster_body():
    # Introduction
    st.write("# Cluster Analysis")
    st.write("## Summary")
    st.write(
        "We removed the season feature and trained a K-Means model. Then"
        " we trained a classification algorithm on the data with season "
        "added back in. This gave us an initial profile and indicated "
        "which features were most important to our clustering. Removing "
        "all other features, we retrained the K-Means model, and went "
        "through the classification process again. This gave us a profile"
        " of the different clusters"
    )
    st.info(
        "We will only concern ourselves with the final clustering and "
        "classification models."
    )
    st.success(
        "This cluster analysis addresses our third Business requirement"
        " and validates Hypothesis 4."
    )

    # Load Data
    cluster_dir = "outputs/ml_pipeline/era_clusters/v1"
    cluster_pipe_v1 = joblib.load(filename=cluster_dir + "/cluster_pipeline.pkl")
    clf_pipe = joblib.load(filename=cluster_dir + "/classification_pipeline.pkl")
    train_dir = "datasets/train/clustering"
    test_dir = "datasets/test/clustering"
    X_TrainSet = get_df("X_TrainSet", train_dir)
    y_TrainSet = get_df("y_TrainSet", train_dir)
    X_TestSet = get_df("X_TestSet", test_dir)
    y_TestSet = get_df("y_TestSet", test_dir)
    season_by_cluster = get_df("season_by_cluster", cluster_dir)
    clusters_profile = get_df("clusters_profile", cluster_dir)
    clusters_profile.set_index("Cluster", inplace=True)
    im_dir = BASE_DIR + cluster_dir
    silhouette_img = plt.imread(im_dir + "/clusters_silhouette.png")
    full_data = get_df("game_w_clusters", cluster_dir)

    # Cluster Profiles
    st.write("## Cluster Profiles")
    st.write(
        "We used an Adaptive Boost Classifier with the hyperparameters "
        "from we used in the previous classification problem. Our "
        "classification model performed exceedingly well, with an "
        "accuracy of 96%."
    )
    # performance report
    st.write("\n")
    st.write(
        "By examining the distribution of the features across each "
        "cluster, we can form a profile for each cluster."
    )
    st.table(clusters_profile)
    st.write("\n")
    st.info(
        "The only features that do not have a significant overlap between"
        " the clusters are 3 point attempts and season. This suggests the "
        "following cluster profiles."
    )
    small_profile = clusters_profile.filter(["fg3a_home", "fg3a_away", "season"])
    st.table(small_profile)
    st.success(
        "Therefore the cleanest profiles of the clusters are as follows"
        ":\n"
        "* **Cluster 0**: Both teams took a moderate number of 3 point "
        "shots and the games occurred between the 1997-1998 and "
        "2009-2010 season.\n"
        "* **Cluster 1**: Both teams took a small number of 3 point "
        "shots and the games occurred between the 1988-1989 and "
        "1999-2000 season.\n"
        "* **Cluster 2**: Both teams took a large number of 3 point "
        "shots and the games occurred between the 2014-2015 and 2019-2020and "
        "2009-2010 season.\n\n"
        "This satisfies our third Business requirement and validates "
        "Hypothesis 4."
    )
    # Feature Distributions
    st.write("## Feature Distribution")
    st.write("### Distribution of Seasons")
    st.write(
        "In the above profiles, we saw that the seasons of each cluster "
        "had minimal overlap. This is is reflected in the distributions."
    )
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 5))
        cluster = season_by_cluster.query(f"Clusters == {i}")
        sns.countplot(data=cluster, x="season").set_title(f"Old Cluster {i}")

        plt.title(f"Season distribution for Cluster {i}")
        st.pyplot(fig)
    st.info("This is a visual representation of what we saw in the profiles "
            "above."
    )
    # distributions of seasons

    st.write("### Distribution of Features by Cluster")
    st.write(
        "We will now look at the distribution of the other important "
        "features across each cluster."
    )
    st.write("\n")
    features = list(clusters_profile.columns)
    feature_pairs = gen_feature_pairs(features)
    pair = st.selectbox("Feature", features, index=0)
    feature_distribution_by_cluster(pair, full_data)
    fig, axes = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=data, x=feature_1, y=feature_2, ax=axes)
    plt.title(
        f"{disp(feature_1)} vs. {disp(feature_2)}: " f"{corr_df[feature_1][feature_2]}"
    )
    st.pyplot(fig)

    st.write("## Pipeline")
    st.write("### Silhouette scores")
    st.write(
        " We used the Elbow method and Silhouette scores to determine "
        "that 3 clusters was optimal."
    )
    st.write("\n")
    st.image(silhouette_img)
    st.write("\n")
    st.write("### Cluster Pipeline Steps")
    st.write(cluster_pipe_v1)
    st.write("\n")
    st.write("### Classification Pipeline Steps")
    st.write(clf_pipe)
    """
    # load cluster analysis files and pipeline
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png")
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )

    # dataframe for cluster_distribution_per_variable()
    df_churn_vs_clusters = load_telco_data().filter(['Churn'], axis=1)
    df_churn_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score is 0.68"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    cluster_distribution_per_variable(df=df_churn_vs_clusters, target='Churn')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    st.write("#### Cluster Profile")
    statement = (
        f"* Historically, **users in Clusters 0 do not tend to Churn**, "
        f"whereas in **Cluster 1 a third of users churned**, "
        f"and in **Cluster 2 a quarter of users churned**. \n"
        f"* From the Predict Churn study, we noticed that the ContractType and InternetService "
        f"are the predictor variables to determine, if a person will churn or not.\n"
        f"* **One potential action** when you detect that a given prospect is expected to churn and "
        f"will belong to cluster 1 or 2 is to mainly avoid month to month contract type, "
        f"like we learned in the churned customer study. \n"
        f"* The salesperson would have then to consider the current product and services "
        f"plan availability and encourage the prospect to move to another contract."
    )
    st.info(statement)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* Cluster 0 has users without internet, who are low spenders with a phone.\n"
        f"* Cluster 1 has users with Internet, who are high spenders with a phone.\n"
        f"* Cluster 2 has users with Internet, who are mid spenders without a phone."
    )
    st.success(statement)

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)


# code coped from "07 - Modeling and Evaluation - Cluster Sklearn" notebook - under "Cluster Analysis" section
def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.value_counts(["Clusters", target]).reset_index()
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"#### Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .groupby(level=0)
                   .apply(lambda x:  100*x / x.sum())
                   .reset_index()
                   .sort_values(by=['Clusters'])
                   )
    df_relative.columns = ['Clusters', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)
"""


def gen_feature_pairs(best_features):
    feature_pairs = []
    feature_stems = list({term.split("_")[0] for term in best_features})
    for stem in feature_stems:
        home_stat = stem + "_home"
        away_stat = stem + "_away"
        if home_stat in best_features and away_stat in best_features:
            feature_pairs.append((home_stat, away_stat))
        elif home_stat in best_features:
            feature_pairs.append((home_stat, ""))
        elif away_stat in best_features:
            feature_pairs.append(("", away_stat))
    return feature_pairs


def feature_distribution_by_cluster(feature_pair, df):
    home, away = feature_pair
    if home and away:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        sns.histplot(
            data=df,
            x=home,
            kde=True,
            element="step",
            ax=axes[0],
            hue="Clusters",
            palette="deep",
        )
        sns.histplot(
            data=df,
            x=away,
            kde=True,
            element="step",
            ax=axes[1],
            hue="Clusters",
            palette="deep",
        )
        axes[0].set_title(f"{home}")
        axes[1].set_title(f"{away}")
        st.pyplot(fig)
    elif home:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        sns.histplot(
            data=df,
            x=home,
            kde=True,
            element="step",
            ax=axes,
            hue="Clusters",
            palette="deep",
        )
        axes.set_title(f"{home}")
        st.pyplot(fig)
    elif away:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        sns.histplot(
            data=df,
            x=away,
            kde=True,
            element="step",
            ax=axes,
            hue="Clusters",
            palette="deep",
        )
        axes.set_title(f"{away}")
        st.pyplot(fig)
