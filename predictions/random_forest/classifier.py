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
#                                     Build data frame using predictors and target             
##################################################################################################################
def build_data(
    csv_files:list[str],
    column_names:list[str],
    remove_rows_with_nan:bool
)->Any:
    print("\n")
    print("Reading and processing data")
    print("Using files:", csv_files)

    # Read each CSV file into a DataFrame and store them in a list
    data = [pd.read_csv(file) for file in csv_files]
    # Concatenate all DataFrames into a single DataFrame
    data = pd.concat(data, ignore_index=True)

    # filter columns containing only predictors and target
    data = data[column_names]

    if remove_rows_with_nan:
        # count nan rows
        nan_rows = data[data.isna().any(axis=1)]
        nan_rows_count = nan_rows.shape[0]
        if nan_rows_count>0: 
            nan_columns = data.columns[data.isna().any()].tolist()
            print("WARNING!")
            print(f"For columns {column_names} there are {nan_rows_count} rows containing nan values")
            print(f"columns containing nan values:{nan_columns}")
            print("Removing rows with nan values")
            data = data.dropna()

    #areas = set(data["area"])
    #discipline = set(data["discipline"])

    # classify using subdiscipline??
    # features["speciality"].value_counts()
    #   subdiscipline
    #   INVESTIGACION EN SALUD                         572
    #   SALUD PUBLICA                                  124
    #   OTRAS ESPECIALIDADES EN MATERIA DE BIOLOGIA      1
    #   OTRAS                    

    # features["speciality"].value_counts()
    print('data set shape is (rows,columns):', data.shape)
    # Show first 5 rows of the data
    #print(data.head(5))

    return data

##################################################################################################################
#                                                  Features (predictors) and label (target)
##################################################################################################################
def extract_predictors_and_target(
    data:Any,
    predictors_names:list[str],
    predictors_dummies_names:list[str],
    target_name:str
)->tuple[Any,Any,list[str],dict[Any,str],dict[str,Any]]:

    print("Extracting features and label")
    # Map categories of the label to integer values
    values = data[target_name].unique()
    values.sort()
    val_index_dic = {value:index for index, value in enumerate(values, start=1)}
    index_val_dic =  {index: value for value, index in val_index_dic.items()}

    # Extract labels (dependent variables/ values to predict) from data
    target = data[target_name].map(lambda value:val_index_dic[value])
    target = np.array(target)

    # Extract features dataframe 
    predictors_df= data[predictors_names]

    # Apply one-hot encoding for categorical predictors
    dummies_names = [name for name in predictors_dummies_names if name in predictors_names]
    if len(dummies_names) > 0:
        predictors_df = pd.get_dummies(predictors_df,columns=dummies_names)
    #predictors_df.head(5)

    # Save features names
    predictors_names = list(predictors_df.columns)

    # Convert features dataframe to numpy array
    predictors = np.array(predictors_df)
    print("\n")
    return predictors, target, predictors_names, index_val_dic, val_index_dic

##################################################################################################################
#                                     Build the training set and testing set
##################################################################################################################
def build_training_sets(
    predictors:Any,
    target:Any,
    random_state:int,
)->tuple:
    print("Building testing and training sets")
    # Split the data into training and testing sets using Skicit-learn
    train_predictors, test_predictors, train_target, test_target = train_test_split(predictors, target, test_size = 0.25,
                                                                            random_state = random_state)

    # Print feature and labels set shapes (total of rows and columns for each set)
    print('Shape of testing features:', test_predictors.shape)
    print('Shape of testing labels:', test_target.shape)
    print('Shape of training features:', train_predictors.shape)
    print('Shape of training labels:', train_target.shape)
    print("\n")

    return train_predictors, test_predictors, train_target, test_target


##################################################################################################################
#                                     Random forest training
##################################################################################################################
def train_random_forest_classifier(
    train_predictors:Any,
    train_target:Any,
    random_state:int
)->Any:    
    # Create a random forest model
    rfc = RandomForestClassifier(n_estimators= 1000, random_state=random_state)

    # Train the random forest model using the training sets (features and labels)
    print("Training model...")
    rfc.fit(train_predictors, np.ravel(train_target))

    print("Model trained")
    print("\n")

    return rfc


##################################################################################################################
#                                     Predictions using test data
##################################################################################################################
def map_values_to_names(
   values:list[Any],
   val_names_val_dic:dict[Any,str]
)->list[str]:
    return [val_names_val_dic[value] for value in values]

def print_classification_info(
    predictions:Any,
    test_target:Any
)->Any:
    # Compute absolute errors
    #errors = abs(predictions - test_labels)
    # Show the "Mean Absolute Error" (MAE)
    #print('Mean Absolute Error:', round(np.mean(errors), 2))
    # Compute "Mean Absolute Percentage Error" (MAPE)
    #mape = 100 * (errors / test_labels)
    # Compute accuracy
    #accuracy = 100 - np.mean(mape)

    """
    NEXUS REFERENCES:
    Confusion Matrix:
    [[ 2  4  0  1  0  0]
    [ 0 20  3  1  0  6]
    [ 1  5  7  0  0  0]
    [ 0  1  0  9  3  2]
    [ 0  7  0  4  3  1]
    [ 0  5  0  3  0  7]] 

                precision    recall  f1-score   support

            0       0.67      0.29      0.40         7
            1       0.48      0.67      0.56        30
            2       0.70      0.54      0.61        13
            3       0.50      0.60      0.55        15
            4       0.50      0.20      0.29        15
            5       0.44      0.47      0.45        15

        accuracy                           0.51        95
    macro avg       0.55      0.46      0.47        95
    weighted avg       0.52      0.51      0.49        95
    """ 
    #accuracy = accuracy_score(test_labels, predictions)
    #print(f"Accuracy:{accuracy:.2f}")
    #print("\n")

    print('Confusion Matrix:\n ', confusion_matrix(test_target,predictions),'\n')
    print(classification_report(y_true=test_target,y_pred=predictions,output_dict=True),'\n')


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
