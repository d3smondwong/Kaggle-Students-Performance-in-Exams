import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# save_object() will be used in data_transformation.py and model_trainer.py to save data_transformer_object and best_model into a pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
# evaluate_models() will be used in model_trainer.py
def evaluate_models(X_train_em, y_train_em,X_test_em,y_test_em,models,param):
    try:
        report = {}

        # loop to interate through the different models to retrieve the model object and corresponding params defined in the list at model_trainer.py
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            # GridSearchCV object is created using the model, parameters, and 3-fold cross-validation (cv=3). 
            # This performs an exhaustive search over the provided parameter grid to find the best hyperparameter combination that minimizes the loss function
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train_em,y_train_em) # The gs.fit(X_train_em, y_train_em) line fits the model using grid search on the training data (X_train_em, y_train_em).

            model.set_params(**gs.best_params_) # The model.set_params(**gs.best_params_) line sets the model's parameters to the best combination found by grid search.
            model.fit(X_train_em,y_train_em) # The model is then refitted on the entire training data using these best parameters (model.fit(X_train_em, y_train_em)) to potentially improve its performance.

            # use the fitted model on both train and test sets to make predictions
            y_train_pred_em = model.predict(X_train_em)

            y_test_pred_em = model.predict(X_test_em)

            # Determine the r2 score to evluate the model performance
            train_model_score = r2_score(y_train_em, y_train_pred_em)

            test_model_score = r2_score(y_test_em, y_test_pred_em)

            # Store test model score in the report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# to load the object via a pickle file. read in binary "rb" mode     
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
  
