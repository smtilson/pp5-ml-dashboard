import os

home_dir = '/workspace/pp5-ml-dashboard'
os.chdir(home_dir)
current_dir = os.getcwd()
print(current_dir)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from src.utils import get_df, save_df



from sklearn.preprocessing import StandardScaler
from feature_engine import transformation as vt
from feature_engine.selection import DropFeatures, SmartCorrelatedSelection
from sklearn.pipeline import Pipeline


# Constants needed for feature engineering
META = ['season', 'play_off']
THRESH = 0.85
TRANSFORMS = {'log_e':(vt.LogTransformer, False), 
                'log_10':(vt.LogTransformer,'10'),
                'reciprocal':(vt.ReciprocalTransformer,False), 
                'power':(vt.PowerTransformer,False),
                'box_cox':(vt.BoxCoxTransformer,False),
                'yeo_johnson':(vt.YeoJohnsonTransformer,False)}
TRANSFORM_ASSIGNMENTS = {
    'yeo_johnson': ['dreb_away', 'blk_home', 'oreb_away', 'fta_away', 'dreb_home', 
                    'ast_home', 'stl_away', 'pts_away', 'stl_home', 'reb_away',
                    'pts_home', 'fgm_away', 'oreb_home', 'pf_away', 'pf_home'],
    'box_cox': ['ast_away', 'fta_home']
                            }


def pipeline(to_drop=None,thresh=THRESH, 
             transform_assignments=TRANSFORM_ASSIGNMENTS):
    if not to_drop:
        to_drop = META
    else:
        to_drop.extend(META)
    pipeline = Pipeline([
        ('dropper', DropFeatures(features_to_drop=to_drop)),
        ('corr_selector', SmartCorrelatedSelection(method="pearson",
                                                   threshold=thresh,
                                                   selection_method="variance",))
                        ])
    for transform in transform_assignments:
        pipeline.steps.append(
            (transform, TRANSFORMS[transform][0](variables=transform_assignments[transform]))
        )
    pipeline.steps.append(('scaler', StandardScaler()))
    return pipeline

def dumb_pipeline(to_drop=None,thresh=THRESH, 
             transform_assignments=TRANSFORM_ASSIGNMENTS):
    if not to_drop:
        to_drop = META
    else:
        to_drop.extend(META)
    pipeline = Pipeline([
        ('dropper', DropFeatures(features_to_drop=to_drop)),
        ('corr_selector', SmartCorrelatedSelection(method="pearson",threshold=thresh,selection_method="variance")),
        ('scaler', StandardScaler())])
    return pipeline

from sklearn.feature_selection import SelectFromModel

# ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

MODELS = {
    'LogisticRegression': LogisticRegression,
    'DecisionTree': DecisionTreeClassifier,
    'RandomForest': RandomForestClassifier,
    'GradientBoosting': GradientBoostingClassifier,
    'ExtraTrees': ExtraTreesClassifier,
    'AdaBoost': AdaBoostClassifier,
    'XGBoost': XGBClassifier
}


def add_feat_selection_n_model(pipeline,model,random_state=42):
    actual_model = MODELS[model]
    pipeline.steps.append(("feat_selection", SelectFromModel(actual_model(random_state=random_state))))
    pipeline.steps.append((model, actual_model(random_state=random_state)))
    return pipeline



from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report, confusion_matrix

def confusion_matrix_and_report(X,y,pipeline,label_map):

  prediction = pipeline.predict(X)

  print('---  Confusion Matrix  ---')
  print(pd.DataFrame(confusion_matrix(y_pred=prediction, y_true=y),
        columns=[ ["Actual " + sub for sub in label_map] ], 
        index= [ ["Prediction " + sub for sub in label_map ]]
        ))
  print("\n")


  print('---  Classification Report  ---')
  print(classification_report(y, prediction, target_names=label_map),"\n")


def clf_performance(X_train,y_train,X_test,y_test,pipeline,label_map):
  print("#### Train Set #### \n")
  confusion_matrix_and_report(X_train,y_train,pipeline,label_map)

  print("#### Test Set ####\n")
  confusion_matrix_and_report(X_test,y_test,pipeline,label_map)

 



training_results = """[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.0s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.001, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=0.001, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 3/5] END model__C=0.001, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=0.001, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 1/5] END model__C=0.001, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=0.001, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 3/5] END model__C=0.001, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   2.1s
[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   2.2s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   2.2s
[CV 1/5] END model__C=0.001, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   0.8s
[CV 5/5] END model__C=0.001, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=0.001, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=0.001, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   0.8s
[CV 5/5] END model__C=0.001, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.0s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   0.9s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=0.01, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=0.01, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.834) precision: (test=0.833) total time=   6.9s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.847) precision: (test=0.848) total time=   7.2s
[CV 1/5] END model__C=0.01, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 5/5] END model__C=0.01, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 4/5] END model__C=0.001, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.5s
[CV 2/5] END model__C=0.001, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   7.8s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.834) precision: (test=0.833) total time=   8.0s
[CV 4/5] END model__C=0.001, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   6.3s
[CV 3/5] END model__C=0.01, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=sag; accuracy: (test=0.834) precision: (test=0.834) total time=   8.7s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.847) precision: (test=0.848) total time=   8.7s
[CV 1/5] END model__C=0.01, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=sag; accuracy: (test=0.847) precision: (test=0.848) total time=   8.9s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.834) precision: (test=0.833) total time=   9.0s
[CV 5/5] END model__C=0.01, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 2/5] END model__C=0.001, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.9s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.847) precision: (test=0.848) total time=   9.1s
[CV 5/5] END model__C=0.01, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=0.001, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   7.9s
[CV 3/5] END model__C=0.01, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 3/5] END model__C=0.01, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 1/5] END model__C=0.01, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.001, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   9.5s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.854) precision: (test=0.871) total time=   7.4s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.847) precision: (test=0.863) total time=   7.7s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.847) precision: (test=0.863) total time=   7.3s
[CV 5/5] END model__C=0.01, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.847) precision: (test=0.863) total time=   8.3s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.854) precision: (test=0.871) total time=   8.0s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.854) precision: (test=0.871) total time=   8.2s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=0.1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.0s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.001, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.2s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=0.1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=0.001, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.5s
[CV 2/5] END model__C=0.01, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   7.9s
[CV 1/5] END model__C=0.1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=sag; accuracy: (test=0.847) precision: (test=0.863) total time=   9.2s
[CV 5/5] END model__C=0.1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=0.01, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   8.3s
[CV 3/5] END model__C=0.1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=sag; accuracy: (test=0.854) precision: (test=0.871) total time=   9.3s
[CV 5/5] END model__C=0.1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   7.9s
[CV 3/5] END model__C=0.1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=0.1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.01, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.4s
[CV 5/5] END model__C=0.1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 4/5] END model__C=0.01, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.7s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   6.4s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   9.0s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.867) total time=   7.6s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.867) total time=   7.1s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.0s
[CV 1/5] END model__C=1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.0s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.4s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   8.0s
[CV 3/5] END model__C=1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.01, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.2s
[CV 5/5] END model__C=1, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.867) total time=  10.0s
[CV 3/5] END model__C=1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=   8.9s
[CV 3/5] END model__C=1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.1, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   8.6s
[CV 1/5] END model__C=1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 5/5] END model__C=1, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=sag; accuracy: (test=0.848) precision: (test=0.867) total time=   9.4s
[CV 2/5] END model__C=0.1, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.5s
[CV 2/5] END model__C=0.1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   7.8s
[CV 1/5] END model__C=1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=0.1, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.5s
[CV 2/5] END model__C=0.1, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   9.6s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   6.8s
[CV 3/5] END model__C=1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 4/5] END model__C=0.1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   8.7s
[CV 5/5] END model__C=1, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.1, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=   9.0s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   8.7s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   2.1s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   7.9s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.4s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.9s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   8.0s
[CV 2/5] END model__C=0.1, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.4s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=10, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=10, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=1, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   7.6s
[CV 4/5] END model__C=1, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.6s
[CV 1/5] END model__C=10, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=10, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 5/5] END model__C=10, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=10, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=10, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 2/5] END model__C=1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   7.7s
[CV 1/5] END model__C=10, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 3/5] END model__C=10, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=1, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.8s
[CV 4/5] END model__C=1, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.6s
[CV 1/5] END model__C=10, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   2.1s
[CV 5/5] END model__C=10, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 4/5] END model__C=1, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   8.1s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 5/5] END model__C=10, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   2.7s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  11.1s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  11.7s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   7.3s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.6s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=1, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=   9.5s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.6s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   8.0s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.8s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   9.4s
[CV 4/5] END model__C=1, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.3s
[CV 1/5] END model__C=100, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 5/5] END model__C=100, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=100, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=10, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   8.1s
[CV 1/5] END model__C=100, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=100, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=100, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=100, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=100, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=10, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.6s
[CV 4/5] END model__C=10, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   9.2s
[CV 1/5] END model__C=100, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 3/5] END model__C=100, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   2.8s
[CV 3/5] END model__C=100, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=10, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   9.2s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.5s
[CV 5/5] END model__C=100, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.5s
[CV 4/5] END model__C=10, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   9.5s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=10, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   9.3s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=10, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=   9.1s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   8.6s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   9.1s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.3s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.9s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=1000, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   9.5s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 4/5] END model__C=10, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  11.7s
[CV 1/5] END model__C=1000, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=100, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.6s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   9.4s
[CV 5/5] END model__C=1000, model__penalty=None, model__solver=lbfgs; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1000, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=1000, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=100, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   8.2s
[CV 5/5] END model__C=1000, model__penalty=None, model__solver=newton-cg; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=100, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.3s
[CV 1/5] END model__C=1000, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=1000, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=1000, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=1000, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=100, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   8.3s
[CV 3/5] END model__C=1000, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.7s
[CV 2/5] END model__C=100, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   8.8s
[CV 1/5] END model__C=0.001, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=1000, model__penalty=None, model__solver=sag; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 4/5] END model__C=100, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   9.6s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  11.6s
[CV 3/5] END model__C=0.001, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   7.8s
[CV 5/5] END model__C=0.001, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.7s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 1/5] END model__C=0.01, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.0s
[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 3/5] END model__C=0.01, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=0.01, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.1s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=100, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  10.9s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   8.6s
[CV 4/5] END model__C=100, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.6s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   9.6s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   2.3s
[CV 1/5] END model__C=0.1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 5/5] END model__C=0.1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=0.1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   8.9s
[CV 4/5] END model__C=1000, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.855) precision: (test=0.873) total time=   7.8s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=1000, model__penalty=None, model__solver=lbfgs; accuracy: (test=0.848) precision: (test=0.868) total time=   8.4s
[CV 3/5] END model__C=1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   0.9s
[CV 4/5] END model__C=1000, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.855) precision: (test=0.873) total time=   8.0s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=   9.6s
[CV 2/5] END model__C=1000, model__penalty=None, model__solver=newton-cg; accuracy: (test=0.848) precision: (test=0.868) total time=   8.6s
[CV 5/5] END model__C=1, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=1000, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.855) precision: (test=0.873) total time=   7.8s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=  11.0s
[CV 2/5] END model__C=1000, model__penalty=None, model__solver=newton-cholesky; accuracy: (test=0.848) precision: (test=0.868) total time=   8.8s
[CV 2/5] END model__C=0.001, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.840) precision: (test=0.861) total time=   7.7s
[CV 5/5] END model__C=10, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 1/5] END model__C=10, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 4/5] END model__C=0.001, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.844) precision: (test=0.868) total time=   7.5s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=10, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 2/5] END model__C=1000, model__penalty=None, model__solver=sag; accuracy: (test=0.848) precision: (test=0.868) total time=   9.4s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=100, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=100, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.862) total time=   8.3s
[CV 5/5] END model__C=100, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.854) precision: (test=0.873) total time=   7.7s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.840) precision: (test=0.850) total time=   9.2s
[CV 4/5] END model__C=1000, model__penalty=None, model__solver=sag; accuracy: (test=0.855) precision: (test=0.873) total time=  10.6s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 4/5] END model__C=0.01, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.854) precision: (test=0.873) total time=   8.4s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=1000, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=0.01, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.849) precision: (test=0.868) total time=   8.9s
[CV 3/5] END model__C=1000, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.865) total time=   8.7s
[CV 5/5] END model__C=1000, model__penalty=l1, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=liblinear; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=0.001, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=0.001, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.0s
[CV 1/5] END model__C=0.001, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.1, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.9s
[CV 5/5] END model__C=0.001, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=0.001, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.5s
[CV 5/5] END model__C=0.001, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.001, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.1, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=  10.3s
[CV 2/5] END model__C=1, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.7s
[CV 4/5] END model__C=1, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.4s
[CV 5/5] END model__C=0.001, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.001, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.5s
[CV 3/5] END model__C=0.001, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.9s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.4s
[CV 4/5] END model__C=10, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   7.7s
[CV 3/5] END model__C=0.001, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=0.01, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   7.8s
[CV 2/5] END model__C=10, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.5s
[CV 5/5] END model__C=0.001, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 3/5] END model__C=0.01, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.9s
[CV 5/5] END model__C=0.01, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=0.01, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   7.7s
[CV 3/5] END model__C=0.01, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=0.01, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.01, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=0.01, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 2/5] END model__C=100, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   9.2s
[CV 4/5] END model__C=100, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.7s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.3s
[CV 2/5] END model__C=1000, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.2s
[CV 5/5] END model__C=0.01, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=0.01, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=1000, model__penalty=l1, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.0s
[CV 1/5] END model__C=0.1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=0.01, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   2.0s
[CV 5/5] END model__C=0.01, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 3/5] END model__C=0.1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.848) precision: (test=0.868) total time=   8.4s
[CV 1/5] END model__C=0.1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=0.1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=liblinear; accuracy: (test=0.855) precision: (test=0.873) total time=   8.7s
[CV 5/5] END model__C=0.1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=0.1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.7s
[CV 1/5] END model__C=0.1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=0.1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   0.9s
[CV 2/5] END model__C=0.001, model__penalty=l1, model__solver=saga; accuracy: (test=0.836) precision: (test=0.830) total time=   9.1s
[CV 5/5] END model__C=0.1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=0.001, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   7.5s
[CV 4/5] END model__C=0.001, model__penalty=l1, model__solver=saga; accuracy: (test=0.846) precision: (test=0.843) total time=   9.5s
[CV 2/5] END model__C=0.001, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   8.2s
[CV 1/5] END model__C=0.1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=0.001, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   7.3s
[CV 2/5] END model__C=0.001, model__penalty=l2, model__solver=saga; accuracy: (test=0.834) precision: (test=0.833) total time=   9.4s
[CV 3/5] END model__C=0.1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 5/5] END model__C=0.1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.001, model__penalty=l2, model__solver=saga; accuracy: (test=0.847) precision: (test=0.848) total time=   9.8s
[CV 3/5] END model__C=1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 1/5] END model__C=1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 5/5] END model__C=1, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=0.001, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   9.3s
[CV 3/5] END model__C=1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=l1, model__solver=saga; accuracy: (test=0.849) precision: (test=0.866) total time=   8.8s
[CV 5/5] END model__C=1, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=0.01, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.3s
[CV 3/5] END model__C=1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 5/5] END model__C=1, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 3/5] END model__C=1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=0.01, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   7.1s
[CV 4/5] END model__C=0.01, model__penalty=l2, model__solver=saga; accuracy: (test=0.854) precision: (test=0.871) total time=   8.3s
[CV 4/5] END model__C=0.01, model__penalty=l1, model__solver=saga; accuracy: (test=0.854) precision: (test=0.871) total time=   9.4s
[CV 5/5] END model__C=1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=0.01, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   8.4s
[CV 1/5] END model__C=10, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.01, model__penalty=l2, model__solver=saga; accuracy: (test=0.847) precision: (test=0.863) total time=   8.9s
[CV 3/5] END model__C=10, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   0.9s
[CV 5/5] END model__C=10, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 1/5] END model__C=10, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 5/5] END model__C=10, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 2/5] END model__C=0.1, model__penalty=l2, model__solver=saga; accuracy: (test=0.848) precision: (test=0.867) total time=   7.7s
[CV 1/5] END model__C=10, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 2/5] END model__C=0.01, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   9.9s
[CV 3/5] END model__C=10, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 2/5] END model__C=0.1, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   7.1s
[CV 4/5] END model__C=0.1, model__penalty=l1, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   8.8s
[CV 5/5] END model__C=10, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=10, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=10, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 2/5] END model__C=0.1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.4s
[CV 4/5] END model__C=0.1, model__penalty=l2, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   8.6s
[CV 4/5] END model__C=0.1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.4s
[CV 3/5] END model__C=10, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=10, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 2/5] END model__C=0.1, model__penalty=l1, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   9.9s
[CV 1/5] END model__C=100, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=100, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 1/5] END model__C=100, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.1s
[CV 3/5] END model__C=100, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=100, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=100, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=100, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=0.1, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   9.2s
[CV 3/5] END model__C=100, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 4/5] END model__C=1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.1s
[CV 2/5] END model__C=1, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.5s
[CV 5/5] END model__C=100, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=100, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.8s
[CV 3/5] END model__C=100, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 1/5] END model__C=1000, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.3s
[CV 5/5] END model__C=100, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=1000, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 1/5] END model__C=1000, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 4/5] END model__C=1, model__penalty=l1, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   9.7s
[CV 2/5] END model__C=1, model__penalty=l2, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   9.4s
[CV 4/5] END model__C=1, model__penalty=l2, model__solver=saga; accuracy: (test=0.854) precision: (test=0.873) total time=   9.2s
[CV 5/5] END model__C=1000, model__penalty=l1, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 4/5] END model__C=1, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   9.0s
[CV 5/5] END model__C=1000, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 3/5] END model__C=1000, model__penalty=l2, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 1/5] END model__C=1000, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 5/5] END model__C=1000, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.2s
[CV 2/5] END model__C=1, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=  10.0s
[CV 2/5] END model__C=1, model__penalty=l1, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=  11.4s
[CV 1/5] END model__C=1000, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.4s
[CV 3/5] END model__C=1000, model__penalty=None, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.6s
[CV 4/5] END model__C=10, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   6.8s
[CV 2/5] END model__C=10, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   7.2s
[CV 5/5] END model__C=1000, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   0.8s
[CV 4/5] END model__C=10, model__penalty=l1, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   8.9s
[CV 4/5] END model__C=10, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   8.1s
[CV 2/5] END model__C=10, model__penalty=l2, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   8.9s
[CV 3/5] END model__C=1000, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   1.5s
[CV 4/5] END model__C=100, model__penalty=l1, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   6.7s
[CV 2/5] END model__C=10, model__penalty=l1, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=  10.1s
[CV 2/5] END model__C=100, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   5.4s
[CV 2/5] END model__C=10, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   8.8s
[CV 4/5] END model__C=100, model__penalty=l2, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   6.6s
[CV 4/5] END model__C=10, model__penalty=l2, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   9.4s
[CV 2/5] END model__C=100, model__penalty=l2, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   7.1s
[CV 4/5] END model__C=100, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   5.6s
[CV 4/5] END model__C=100, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   6.5s
[CV 2/5] END model__C=100, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   7.0s
[CV 4/5] END model__C=1000, model__penalty=l2, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   4.2s
[CV 2/5] END model__C=100, model__penalty=l1, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   8.3s
[CV 2/5] END model__C=1000, model__penalty=l1, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   5.7s
[CV 4/5] END model__C=1000, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   2.6s
[CV 4/5] END model__C=1000, model__penalty=l1, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   5.5s
[CV 2/5] END model__C=1000, model__penalty=None, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   3.9s
[CV 2/5] END model__C=1000, model__penalty=l2, model__solver=saga; accuracy: (test=0.848) precision: (test=0.868) total time=   4.7s
[CV 2/5] END model__C=1000, model__penalty=elasticnet, model__solver=saga; accuracy: (test=nan) precision: (test=nan) total time=   3.5s
[CV 4/5] END model__C=1000, model__penalty=None, model__solver=saga; accuracy: (test=0.855) precision: (test=0.873) total time=   3.9s"""

lines = training_results.split('\n')
def find_solver(line) -> str:
    return line.split('model__solver=')[1].split(';')[0].strip()

def find_penalty(line) -> str:
    return line.split('model__penalty=')[1].split(',')[0].strip()

def find_C(line) -> str:
    return line.split('model__C=')[1].split(',')[0].strip()

def sort_lines(lines) -> list:
    return sorted(lines, key=lambda x: (find_solver(x), find_penalty(x), find_C(x)))   

sorted_lines = sort_lines(lines)
