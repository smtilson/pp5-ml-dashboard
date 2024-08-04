# Functions from notebooks for easy import into other notebooks
import pandas as pd
import matplotlib.pyplot as plt


def find_features(X_train, fitted_pipe, initial_drop):
    total = len(X_train.columns)
    corr_dropped = list(fitted_pipe["corr_selector"].features_to_drop_)
    auto_dropped = initial_drop + corr_dropped
    cols = [col for col in X_train.columns if col not in auto_dropped]

    features = fitted_pipe["feat_selection"].get_support()
    X = X_train.filter(cols)
    if len(X.columns) != features.shape[0]:
        raise ValueError
    feat_selected_dropped = []
    feat_selected = []
    for index, col in enumerate(X.columns):
        if not features[index]:
            X.drop(col, axis=1, inplace=True)
            feat_selected_dropped.append(col)
        else:
            feat_selected.append(col)
    dropped = set(auto_dropped + feat_selected_dropped)
    kept = set(X.columns)
    if set.intersection(dropped, kept):
        raise ValueError(str(set.intersection(dropped, kept)))
    missing = [col for col in X_train.columns
               if col not in set.union(dropped, kept)]
    if missing:
        raise ValueError(str(missing))
    if total != len(kept) + len(dropped):
        raise ValueError(str(total - len(kept) - len(dropped)))
    return list(X.columns), auto_dropped + feat_selected_dropped


# Function inspired by similar function for tree based models below which is
# taken from the SciKit learn lessons
def feature_importance_logistic_regression(pipe, step_name,
                                           X_TrainSet, initial_drop):
    coefficients = pipe[step_name].coef_[0]
    features, _ = find_features(X_TrainSet, pipe, initial_drop)
    importance_list = [
        (feature, X_TrainSet[feature].std() * coeff)
        for feature, coeff in zip(features, coefficients)
    ]
    df_feature_importance = pd.DataFrame(
        data={
            "Features": [term[0] for term in importance_list],
            "Importance": [abs(term[1]) for term in importance_list],
        }
    ).sort_values(by="Importance", ascending=False)

    best_features = df_feature_importance["Features"].to_list()

    # Most important features statement and plot
    print(
        f"* These are the {len(best_features)} most important features in "
        f"descending order. "
        f"The model was trained on them: \n"
        f"{df_feature_importance['Features'].to_list()}"
    )

    df_feature_importance.plot(kind="bar", x="Features", y="Importance")

    for feature, coefficient in zip(features, coefficients):
        std = X_TrainSet[feature].std()
        print(feature, std * coefficient)


# This code is from Unit 04 of the Scikit-Learn lessons
def feature_importance_tree_based_models(model, columns):
    df_feature_importance = pd.DataFrame(
        data={"Features": columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    best_features = df_feature_importance["Features"].to_list()

    # Most important features statement and plot
    print(
        f"* These are the {len(best_features)} most important features in "
        f"descending order. "
        f"The model was trained on them: \n"
        f"{df_feature_importance['Features'].to_list()}"
    )

    df_feature_importance.plot(kind="bar", x="Features", y="Importance")
    plt.show()
