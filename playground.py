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

train_dir = 'train/csv'
X_TrainSet = get_df('X_TrainSet',train_dir)
Y_TrainSet = get_df('Y_TrainSet',train_dir)

test_dir = 'test/csv'
X_TestSet = get_df('X_TestSet',test_dir)
Y_TestSet = get_df('Y_TestSet',test_dir)

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

PIPELINES = {}
for model in MODELS:
    base_pipe = pipeline()
    PIPELINES[model] = add_feat_selection_n_model(base_pipe,model)

rand_forest_pipe = pipeline()
rand_forest_pipe.steps.append(('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))))
#X_TrainSet_trans = logistic_pipe.fit_transform(X_TrainSet,Y_TrainSet)
#X_TestSet_trans = logistic_pipe.transform(X_TestSet)
rand_forest_pipe.steps.append(('model',RandomForestClassifier(random_state=42)))

rand_forest_pipe
from sklearn.model_selection import GridSearchCV

param_grid = {"model__n_estimators":[50,20],
              }
GRIDS = {}
pipes_list = list(PIPELINES.values())
pipe1 = pipes_list[0]
grid= GridSearchCV(estimator=pipe1,
                    param_grid={},
                    cv=5,
                    n_jobs=-2,
                    verbose=3,
                    scoring='accuracy')
grid.fit(X_TrainSet,Y_TrainSet)
#GRIDS[pipe] = grid

best_pipe = grid.best_estimator_
res = (pd.DataFrame(grid.cv_results_)
       .sort_values(by='mean_test_score',ascending=False)
       .filter(['params','mean_test_score'])
       .values)

res
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

 

clf_performance(X_train=X_TrainSet, y_train=Y_TrainSet,
                X_test=X_TestSet, y_test=Y_TestSet,
                pipeline=best_pipe,
                label_map= ['win', 'loss'] 
                )
