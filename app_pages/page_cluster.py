# This file follows that of the Churnometer walkthrough project
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from src.utils import get_df, disp, undisp
from src.display import display_report, display_features_tree_based
import os


if os.path.isfile("env.py"):
    import env  # noqa: F401
BASE_DIR = os.environ.get("BASE_DIR")


def page_cluster_body():
    # TOC
    st.write("* [Cluster Profiles](#cluster-profiles)")
    st.write("* [Feature Distribution](#feature-distribution)")
    st.write("* [Cluster Pipeline](#cluster-pipeline)")
    st.write("* [Classification Pipeline](#classification-pipeline)")

    # Introduction
    st.write("# Cluster Analysis")
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
    cluster_pipe = joblib.load(filename=cluster_dir + "/cluster_pipeline.pkl")
    clf_pipe = joblib.load(filename=cluster_dir + "/clf_pipeline.pkl")
    test_dir = "datasets/test/clustering"
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
    st.write(
        "By examining the distribution of the features across each "
        "cluster, we can form a profile for each cluster."
    )
    st.dataframe(clusters_profile)
    st.info(
        "The only features that do not have a significant overlap between"
        " the clusters are 3 point attempts and season. This suggests the "
        "following cluster profiles."
    )
    small_profile = clusters_profile.filter(["fg3a_home", "fg3a_away",
                                             "season"])
    st.dataframe(small_profile)
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
        fig, ax = plt.subplots(figsize=(4, 2))
        cluster = season_by_cluster.query(f"Clusters == {i}")
        sns.countplot(data=cluster, x="season").set_title(f"Old Cluster {i}")

        plt.title(f"Season distribution for Cluster {i}")
        st.pyplot(fig)
    st.info("This is a visual representation of what we saw in the profiles "
            "above.")

    st.write("### Distribution of Features by Cluster")
    st.write(
        "We will now look at the distribution of the other important "
        "features across each cluster."
    )
    features = list(clusters_profile.columns)
    feature_pairs = gen_feature_pairs(features)
    pair = st.selectbox("Feature", feature_pairs, index=0)
    feature_distribution_by_cluster(pair, full_data)

    st.write("## Cluster Pipeline")
    st.write("### Silhouette scores")
    st.write(
        " We used the Elbow method and Silhouette scores to determine "
        "that 3 clusters was optimal."
    )
    st.image(silhouette_img, width=400)
    st.write("### Cluster Pipeline Steps")
    st.write(cluster_pipe)
    st.write("## Classification Pipeline")
    st.write(
        "We trained an Adaptive Boost Classifier to determine which "
        "cluster games belonged to, with season data added back in. This "
        "allowed us to construct profiles of the clusters by looking at "
        "the important features of these classifiers. We then dropped the"
        " unimportant features, retrained the clustering, and refit the "
        "classifier based on the new clustering to arrive at our current "
        "clustering and cluster profiles."
    )
    st.write("### Test Set Performance")
    labels = ["Cluster 0", "Cluster 1", "Cluster 2"]
    display_report(clf_pipe, X_TestSet, y_TestSet, label_map=labels)
    st.write("### Important Features")
    st.write(
        "These are the features that our final clustering and "
        "classifiacation models were trained on."
    )
    display_features_tree_based(clf_pipe, X_TestSet)
    st.write("### Classification Pipeline Steps")
    st.write(clf_pipe)


def gen_feature_pairs(best_features):
    feature_pairs = []
    feature_stems = list({term.split("_")[0] for term in best_features})
    for stem in feature_stems:
        home_stat = stem + "_home"
        away_stat = stem + "_away"
        if home_stat in best_features and away_stat in best_features:
            feature_pairs.append(disp(home_stat) + " vs. " + disp(away_stat))
        elif home_stat in best_features:
            feature_pairs.append(disp(home_stat))
        elif away_stat in best_features:
            feature_pairs.append(disp(away_stat))
    return feature_pairs


def feature_distribution_by_cluster(pair, df):
    if " vs. " in pair:
        home, away = pair.split(" vs. ")
    elif "home" in pair.lower():
        home = pair
        away = False
    elif "away" in pair.lower():
        away = pair
        home = False
    if home and away:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        sns.histplot(
            data=df,
            x=undisp(home),
            kde=True,
            element="step",
            ax=axes[0],
            hue="Clusters",
            palette="deep",
        )
        sns.histplot(
            data=df,
            x=undisp(away),
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
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        sns.histplot(
            data=df,
            x=undisp(home),
            kde=True,
            element="step",
            ax=axes,
            hue="Clusters",
            palette="deep",
        )
        axes.set_title(f"{home}")
        st.pyplot(fig)
    elif away:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        sns.histplot(
            data=df,
            x=undisp(away),
            kde=True,
            element="step",
            ax=axes,
            hue="Clusters",
            palette="deep",
        )
        axes.set_title(f"{away}")
        st.pyplot(fig)
