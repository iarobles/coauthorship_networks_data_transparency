# Pandas is used for data manipulation
from typing import Any
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
#
from sklearn.model_selection import train_test_split
# Import Random forest classifier and random forest regressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



##################################################################################################################
#                                     Random forest training
##################################################################################################################
def train_random_forest_classifier(
    train_predictors:Any,
    train_target:Any,
    random_state:int,
    class_weight:str
)->Any:    
    # Create a random forest model
    rfc = RandomForestClassifier(n_estimators= 1000, random_state=random_state, class_weight=class_weight)
    
    # Train the random forest model using the training sets (features and labels)
    print("Training model...")
    rfc.fit(train_predictors, np.ravel(train_target))

    print("Model trained")
    print("\n")

    return rfc



##################################################################################################################
#                                     Variable Importances
##################################################################################################################

def print_feature_importances(
    features_importances:list[tuple[str,float]]
)->None:
    print("feature importances (filtered)")
    for feat_impt_tuple in  features_importances:
        print(f"Feature name: {feat_impt_tuple[0]},  Importance: {feat_impt_tuple[1]}")

def get_predictors_importances_by_threshold(
    predictor_importances:list[float],
    predictors_names:list[str],
    importance_threshold:float
)->list[tuple[str,float]]:
    # NEXUS REFERENCE:
    # [('Closeness_origin', 0.16), ('Number_origin', 0.15), ('Clustering', 0.14), ('Average_degree', 0.13), 
    # ('Closeness_residence', 0.09), ('Number_residence', 0.09), ('Closeness', 0.08),
    # ('Assortativity', 0.08), ('Betweenness', 0.08)]

    # Extract feature importances values
    #features_importances = list(rf.feature_importances_)

    # Create a list of tuples (importance and its features)
    predictor_importance_list = [(feature_name, round(importance, 3)) for feature_name, importance in zip(predictors_names, predictor_importances)]

    # Sort by importance value in descending order.
    predictor_importance_list = sorted(predictor_importance_list, key = lambda feat_impt_tuple: feat_impt_tuple[1], reverse = True)

    # find features contributing to 80% of importances
    total_importance = 0
    index_importance = 0
    for index_importance,importance_tuple in enumerate(predictor_importance_list):
        feat_name = importance_tuple[0]
        feat_val = importance_tuple[1]
        total_importance = total_importance + feat_val
        if total_importance>=importance_threshold: #threshold
            break

    # print most important features again after filtering:
    predictor_features = predictor_importance_list[0:index_importance+1]
    
    return predictor_features
