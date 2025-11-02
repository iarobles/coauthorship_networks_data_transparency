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

from imblearn.over_sampling import SMOTE

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
    use_smote:bool,
    target_name:str
)->tuple:
    print("Building testing and training sets")
    # Split the data into training and testing sets using Skicit-learn
    train_predictors, test_predictors, train_target, test_target = train_test_split(predictors, target, test_size = 0.25,
                                                                            random_state = random_state)
    
    # print 

    # Apply SMOTE to the training set
    if use_smote:
        print("####### Applying SMOTE to the training set, target name:", target_name,"#########")
        # print total of each class before SMOTE
        unique, counts = np.unique(train_target, return_counts=True)
        print("Training set class distribution before SMOTE:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples")  
        smote = SMOTE(random_state=random_state)
        train_predictors, train_target = smote.fit_resample(train_predictors, train_target)
        # print total of each class after SMOTE
        unique, counts = np.unique(train_target, return_counts=True)
        print("Training set class distribution after SMOTE:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples")

        print("######### finished SMOTE #########")

    # Print feature and labels set shapes (total of rows and columns for each set)
    print('Shape of testing features:', test_predictors.shape)
    print('Shape of testing labels:', test_target.shape)
    print('Shape of training features:', train_predictors.shape)
    print('Shape of training labels:', train_target.shape)
    print("\n")

    return train_predictors, test_predictors, train_target, test_target


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

