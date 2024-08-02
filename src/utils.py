# these are utilities for various repetitive tasks
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from feature_engine import transformation as vt
import warnings

warnings.filterwarnings("ignore")

if os.path.isfile("env.py"):
    import env  # noqa: F401
BASE_DIR = os.environ.get("BASE_DIR")


def get_df(name: str, target_dir) -> pd.DataFrame:
    if "outputs/" not in target_dir:
        target_dir = "outputs/" + target_dir
    if "workspace" not in target_dir:
        target_dir = BASE_DIR + target_dir
    file_path = target_dir + "/" + name + ".csv"
    df = pd.read_csv(file_path)
    if "game_id" in df.columns:
        df.set_index("game_id", inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
        df.set_index("Index", inplace=True)
    return df


def save_df(df, name, target_dir, index=True):
    if "outputs/" not in target_dir:
        target_dir = "outputs/" + target_dir
    if "workspace" not in target_dir:
        target_dir = BASE_DIR + target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    df.to_csv(target_dir + "/" + name + ".csv", index=index)


def disp(feature_name: str) -> str:
    string = feature_name.replace("_", " ")
    words = string.split()
    new_words = [special_caps(word) for word in words]
    return " ".join(new_words)


def undisp(display_name: str) -> str:
    string = display_name.replace(" ", "_")
    return string.lower()


def special_caps(string: str) -> str:
    title_words = {"plus", "home", "away"}
    if len(string) <= 4 and string not in title_words:
        return string.upper()
    else:
        return string.title()


def divide_range(start=0, stop=100, num=4, precision=3):
    length = stop - start
    step = length / num
    initial = [start + step * i for i in range(num + 1)]
    return [round(item, precision) for item in initial]


def count_threshold_changes(df, threshold_list, corr=True):
    """
    df should be a square matrix, like a matrix of correlation coefficients.
    Does this also work for pandas.series?
    """
    trivial = df.shape[0] if corr else 0
    changes = []
    for threshold in threshold_list:
        count = np.count_nonzero(abs(df) > threshold) - trivial
        if not changes or count != changes[-1][0]:
            changes.append((threshold, count))
    return changes


def get_pairs(df, threshold):
    """
    df should be a square matrix, like a matrix of correlation coefficients.
    """
    pairs = []
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                continue
            elif abs(df[col1][col2]) > threshold:
                if (col2, col1, round(df[col1][col2], 3)) not in pairs:
                    pairs.append((col1, col2, round(df[col1][col2], 3)))
    pairs.sort(key=lambda x: x[2])
    return pairs


def add_cat_date(
    df: pd.DataFrame, date_name: str, symbol="-", y_pos=0, m_pos=1, d_pos=2
) -> pd.DataFrame:
    # the date_name argument should be the name of the column containing the
    # date data maybe I should take into account the format

    df["day"] = df.apply(
        lambda x: int(x[date_name].split(symbol)[d_pos].split()[0]), axis=1
    )
    df["month"] = df.apply(lambda x: int(x[date_name].split(symbol)[m_pos]),
                           axis=1)
    df["year"] = df.apply(lambda x: int(x[date_name].split(symbol)[y_pos]),
                          axis=1)
    return df


# The following are from the feature engineering notebook in the walkthrough
# project with minor modifications.
def FeatureEngineeringAnalysis(df):
    """
    - used for quick feature engineering on numerical variables
    to decide which transformation can better transform the distribution shape
    - Once transformed, use a reporting tool, like ydata-profiling, to evaluate
    the distributions
    """
    check_missing_values(df)
    list_column_transformers = [
            "log_e",
            "log_10",
            "reciprocal",
            "power",
            "box_cox",
            "yeo_johnson",
        ]

    # Loop in each variable and engineer the data according to the analysis
    # type
    df_feat_eng = pd.DataFrame([])
    for column in df.columns:
        # create additional columns (column_method) to apply the methods
        df_feat_eng = pd.concat([df_feat_eng, df[column]], axis=1)
        for method in list_column_transformers:
            df_feat_eng[f"{column}_{method}"] = df[column]

        # Apply transformers in respective column_transformers
        df_feat_eng, list_applied_transformers = apply_transformers(
            df_feat_eng, column
        )

        # For each variable, assess how the transformations perform
        transformer_evaluation(
            column, list_applied_transformers, df_feat_eng
        )

    return df_feat_eng


def check_missing_values(df):
    if df.isna().sum().sum() != 0:
        msg = (
            "There is a missing value in your dataset. Please handle that "
            "before getting into feature engineering."
        )
        raise SystemExit(msg)


def apply_transformers(df_feat_eng, column):
    for col in df_feat_eng.select_dtypes(include="category").columns:
        df_feat_eng[col] = df_feat_eng[col].astype("object")

    df_feat_eng, list_applied_transformers = FeatEngineering_Numerical(
        df_feat_eng, column
    )

    return df_feat_eng, list_applied_transformers


def transformer_evaluation(
    column, list_applied_transformers, df_feat_eng
):
    # For each variable, assess how the transformations perform
    print(f"* Variable Analyzed: {column}")
    print(f"* Applied transformation: {list_applied_transformers} \n")
    for col in [column] + list_applied_transformers:
        DiagnosticPlots_Numerical(df_feat_eng, col)
        print("\n")


def DiagnosticPlots_Numerical(df, variable):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.histplot(data=df, x=variable, kde=True, element="step", ax=axes[0])
    stats.probplot(df[variable], dist="norm", plot=axes[1])
    sns.boxplot(x=df[variable], ax=axes[2])

    axes[0].set_title("Histogram")
    axes[1].set_title("QQ Plot")
    axes[2].set_title("Boxplot")
    fig.suptitle(f"{variable}", fontsize=30, y=1.05)
    plt.tight_layout()
    plt.show()


def FeatEngineering_Numerical(df_feat_eng, column):
    list_methods_worked = []

    # LogTransformer base e
    try:
        lt = vt.LogTransformer(variables=[f"{column}_log_e"])
        df_feat_eng = lt.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_log_e")
    except Exception:
        df_feat_eng.drop([f"{column}_log_e"], axis=1, inplace=True)

    # LogTransformer base 10
    try:
        lt = vt.LogTransformer(variables=[f"{column}_log_10"], base="10")
        df_feat_eng = lt.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_log_10")
    except Exception:
        df_feat_eng.drop([f"{column}_log_10"], axis=1, inplace=True)

    # ReciprocalTransformer
    try:
        rt = vt.ReciprocalTransformer(variables=[f"{column}_reciprocal"])
        df_feat_eng = rt.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_reciprocal")
    except Exception:
        df_feat_eng.drop([f"{column}_reciprocal"], axis=1, inplace=True)

    # PowerTransformer
    try:
        pt = vt.PowerTransformer(variables=[f"{column}_power"])
        df_feat_eng = pt.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_power")
    except Exception:
        df_feat_eng.drop([f"{column}_power"], axis=1, inplace=True)

    # BoxCoxTransformer
    try:
        bct = vt.BoxCoxTransformer(variables=[f"{column}_box_cox"])
        df_feat_eng = bct.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_box_cox")
    except Exception:
        df_feat_eng.drop([f"{column}_box_cox"], axis=1, inplace=True)

    # YeoJohnsonTransformer
    try:
        yjt = vt.YeoJohnsonTransformer(variables=[f"{column}_yeo_johnson"])
        df_feat_eng = yjt.fit_transform(df_feat_eng)
        list_methods_worked.append(f"{column}_yeo_johnson")
    except Exception:
        df_feat_eng.drop([f"{column}_yeo_johnson"], axis=1, inplace=True)

    return df_feat_eng, list_methods_worked
