from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def get_best_scores(grid_collection):
  for name, grid in grid_collection.items():
      res = (pd.DataFrame(grid.cv_results_)
         .sort_values(by=['mean_test_precision','mean_test_accuracy'],ascending=False)
         .filter(['params','mean_test_precision','mean_test_accuracy'])
        .values)
      intro = f"Best {name} model:"
      print(f"{intro} Avg. Precision: {res[0][1]*100}%.")
      print(f"{len(intro)*' '} Avg. Accuracy: {res[0][2]*100}%.")
      print()



# Taken from Churnometer Walkthrough project
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

def grid_search_report_best(grid_collection,X_train,Y_train,X_test,Y_test, label_map):
    for name, grid in grid_collection.items():
        best_pipe = grid.best_estimator_
        print(name)
        clf_performance(X_train, Y_train, X_test, Y_test, best_pipe, label_map)
 