from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def get_best_scores(grid_collection):
  for name, grid in grid_collection.items():
    res = (pd.DataFrame(grid.cv_results_)
       .sort_values(by='mean_test_score',ascending=False)
       .filter(['params','mean_test_score'])
       .values)
    print(name)
    print(res)
    print()


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
 