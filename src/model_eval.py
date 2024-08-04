from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    accuracy_score,
)
import pandas as pd
import ast
import math


def get_best_scores(grid_collection):
    for name, grid in grid_collection.items():
        res = (
            pd.DataFrame(grid.cv_results_)
            .sort_values(
                by=["mean_test_precision", "mean_test_accuracy"],
                ascending=False
            )
            .filter(["params", "mean_test_precision", "mean_test_accuracy"])
            .values
        )
        intro = f"Best {name} model:"
        print(f"{intro} Avg. Precision: {res[0][1]*100}%.")
        print(f"{len(intro)*' '} Avg. Accuracy: {res[0][2]*100}%.")
        print()


def get_best_params_df(df, display_num=3):
    res = (
        df.sort_values(
            by=["mean_test_precision", "mean_test_accuracy"], ascending=False
        )
        .filter(["params", "mean_test_precision", "mean_test_accuracy"])
        .values
    )
    print("Best parameters for current model:")
    for i in range(display_num):
        params = res[i][0]
        if isinstance(params, str):
            params = ast.literal_eval(params)
        for key, value in params.items():
            print(f"{key.split('__')[1]}: {value}")
        print(f"Avg. Precision: {res[0][1]*100}%.")
        print(f"Avg. Accuracy: {res[0][2]*100}%.")
        print()


def collect_like_estimators(results_df):
    grouped_dict = {}
    for _, result in results_df.iterrows():
        p_score = result["mean_test_precision"]
        a_score = result["mean_test_accuracy"]
        if math.isnan(p_score):
            p_score = -1
        if math.isnan(a_score):
            a_score = -1
        score = (p_score, a_score)
        if score not in grouped_dict.keys():
            grouped_dict[score] = [result.to_dict()]
        else:
            result_dict = result.to_dict()
            grouped_dict[score].append(result_dict)
    return grouped_dict


def present_score_counts(results_df, display_num=4):
    g_dict = collect_like_estimators(results_df)
    count = 0
    print("---  Score Counts  ---")
    sorted_grouping = {
        k: v
        for k, v in sorted(
            g_dict.items(), reverse=True,
            key=lambda item: (item[0][0], item[0][1])
        )
    }
    for k, v in sorted_grouping.items():
        if count >= display_num:
            break
        print(f"Precision: {k[0]}, Accuracy: {k[1]}")
        print(f"Count: {len(v)}")
        count += 1
        print()
    # return sorted_grouping


def score_stats(results_df):
    grouped_params = collect_like_estimators(results_df)
    print("---  Score Stats  ---")
    most = (0, 0, 0)
    maxim = (0, 0)
    max_p = 0
    max_a = 0
    try:
        for key, value in grouped_params.items():
            if key[0] > max_p:
                max_p = key[0]
            if key[1] > max_a:
                max_a = key[1]
            if len(value) > most[2]:
                most = (key[0], key[1], len(value))
            if key[0] > maxim[0]:
                maxim = key
            elif key[0] == maxim[0] and key[1] > maxim[1]:
                maxim = key
    except TypeError as e:
        print(str(e))
        print(f"{key[0]=}, {key[1]=}, {len(value)=}")
        print(f"{max_a=}, {max_p=}")
        print(f"{most=}, {maxim=}")

    print(
        f'Most Common: Precision: {most[0]}\n{" "*13}Accuracy: {most[1]}'
        f'\n{" "*13}Count: {most[2]}'
    )
    print(
        f'Max Score: Precision: {maxim[0]}\n{" "*11}Accuracy: {maxim[1]}'
        f'\n{" "*11}Count: {len(grouped_params[maxim])}'
    )
    print(f"Max Precision: {max_p}\nMax Accuracy: {max_a}")
    return maxim


def present_param_counts(results_df, score, exclude=None):
    if exclude is None:
        exclude = []
    grouped_estimator_params = collect_like_estimators(results_df)
    best_params = grouped_estimator_params[score]
    param_count = {}
    for param in best_params:
        param_dict = param["params"]
        bad_string = "DecisionTreeClassifier"
        if isinstance(param_dict, str):
            if bad_string in param_dict:
                terms = param_dict.split(", '")
                terms = [term for term in terms if bad_string not in term]
                param_dict = ", '".join(terms)
            try:
                param_dict = ast.literal_eval(param_dict)
            except ValueError as e:
                print(param_dict)
                raise e
        for key, value in param_dict.items():
            if (key, value) in param_count:
                param_count[(key, value)] += 1
            else:
                param_count[(key, value)] = 1
    sorted_count = {
        k: v
        for k, v in sorted(
            param_count.items(), reverse=True,
            key=lambda item: (item[0][0], item[1])
        )
    }
    for key, value in sorted_count.items():
        if key[0] in exclude:
            continue
        print(f"{key[0]}: {key[1]}, Count: {value}")


def evaluate_param_on_test_set(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return ((precision, accuracy), pipe.param_dict)


# Taken from Churnometer Walkthrough project
def confusion_matrix_and_report(X, y, pipeline, label_map):
    prediction = pipeline.predict(X)
    print("---  Confusion Matrix  ---")
    print(
        pd.DataFrame(
            confusion_matrix(y_pred=prediction, y_true=y),
            columns=[["Actual " + sub for sub in label_map]],
            index=[["Prediction " + sub for sub in label_map]],
        )
    )
    print("\n")
    print("---  Classification Report  ---")
    print(classification_report(y, prediction, target_names=label_map), "\n")


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    print("#### Train Set #### \n")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)
    print("#### Test Set ####\n")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)


def gen_clf_report(X, y, pipeline, label_map=None):
    if label_map is None:
        label_map = ["Away Wins", "Home Wins"]
    prediction = pipeline.predict(X)
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_pred=prediction, y_true=y),
        columns=[["Actual " + sub for sub in label_map]],
        index=[["Prediction " + sub for sub in label_map]],
    ).T
    perfomance_dict = classification_report(
        y, prediction, target_names=label_map, output_dict=True
    )
    accuracy_score = perfomance_dict["accuracy"]
    del perfomance_dict["accuracy"]
    performance_report = pd.DataFrame.from_dict(perfomance_dict).T
    return conf_matrix, performance_report, accuracy_score


def grid_search_report_best(
    grid_collection, X_train, y_train, X_test, y_test, label_map
):
    for name, grid in grid_collection.items():
        best_pipe = grid.best_estimator_
        print(name)
        clf_performance(X_train, y_train, X_test, y_test, best_pipe, label_map)
