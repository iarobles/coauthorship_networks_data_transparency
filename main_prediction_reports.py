import os
from typing import Any, Literal, Optional
from pprint import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import predictions.random_forest.classifier as rfc
import filetools.file_utilities as fu
import mathutils.utils as mu
import config_plot_predictions_reports as cppr
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter

##################################################################################################################
#                                                  CONFIGURATION
##################################################################################################################
RANDOM_STATE = 34 
IMPORTANCE_THRESHOLD=0.8

#Y_LABEL = "Porcentaje de Personas en el SNII"
Y_LABEL = "Percent of people in SNII"

# CATEGORY NAMES
CAT_1_NAME = "Category 1"#"Categoría 1"
CAT_2_NAME = "Category 2"#"Categoría 2"
CAT_3_NAME = "Category 3"#"Categoría 3"
CAT_4_NAME = "Category 4"#"Categoría 4"

TARGET_DISCIPLINE = "discipline"
#TARGET_AFFILIATION = "affiliation"
TARGET_AFFILIATION = "affiliation_category"
TARGET_GENDER = "gender"
ALL_TARGETS = [TARGET_DISCIPLINE,TARGET_AFFILIATION,TARGET_GENDER]

VAL_ALIAS_MAPS = {
    TARGET_GENDER:{
        #"alias":"Género",
        "alias":"Gender",
        "val_alias_map": {
            "f":"Female",#"Femenino",
            "m":"Male",#Masculino
        }
    },
    TARGET_DISCIPLINE:{
        "alias":"Discipline", #Disciplina
        "val_alias_map": {
            "CIENCIAS DE LA SALUD":"Health Sciences",#"Ciencias de la salud",
            "CIENCIAS ECONOMICAS":"Economic sciences",#"Ciencias económicas",
            "INGENIERIA":"Engineering"#"Ingeniería"
        }
    },
    TARGET_AFFILIATION:{
        "alias":"Affiliation",#"Afiliación",
        "val_alias_map": {
            CAT_1_NAME:CAT_1_NAME,
            CAT_2_NAME:CAT_2_NAME,
            CAT_3_NAME:CAT_3_NAME,
            CAT_4_NAME:CAT_4_NAME
        }
    }
}

REPORT_DIR = "data/reports"
FIGURE_FORMAT = "eps" # png or jpg or eps
PALETA_COLORES = ['#EB6123','#512888','#dc267f','#648fff']

# IMPORTANT: if you add another CONSTANT for another csv file discipline (like matematicos) then
#            you have to add another entry in CSV_FILE_INFO
HEALTH_CSV_FILENAME = "data/snii_data/salud_ego_included_coco_anonymized.csv"
ENGINEERS_CSV_FILENAME = "data/snii_data/ingenieros_ego_included_coco_anonymized.csv"
ECONOMISTS_CSV_FILENAME = "data/snii_data/economistas_ego_included_coco_anonymized.csv"

CSV_FILE_PATHS = [HEALTH_CSV_FILENAME, ENGINEERS_CSV_FILENAME, ECONOMISTS_CSV_FILENAME]
#CSV_FILES = [HEALTH_CSV_FILENAME]
#CSV_FILES = [ENGINEERS_CSV_FILENAME]
#CSV_FILES = [ECONOMISTS_CSV_FILENAME]

TARGET = TARGET_AFFILIATION


PREDICTORS = [
    # Metrics that are computed for specific nodes (the ego in this case):
    {
        "name": "core_number",
        # "alias": "Número Core"
        "alias": "Core number"
    },
    # {
    #     "name": "degree_centrality",
    #     # "alias": "Centralidad de grado"
    #     "alias": "Degree centrality"
    # },
    {
        "name": "closeness_centrality",
        # "alias": "Centralidad de cercanía"
        "alias": "Closeness centrality"
    },
    {
        "name": "clustering_coefficient",
        # "alias": "Coeficiente de agrupamiento"
        "alias": "Clustering coefficient"
    },
    # 'betweenness_centrality',

    # Metrics that are computed for all graph:
    {
        "name": "average_degree",
        # "alias": "Grado promedio"
        "alias": "Average degree"
    },
    {
        "name": "average_clustering",
        # "alias": "Agrupamiento promedio"
        "alias": "Average clustering"
    },
    # 'diameter',
    {
        "name": "density",
        # "alias": "Densidad"
        "alias": "Density"
    },
    # 'domination_number',
    # 'chromatic_number',
    # 'number_connected_components',
    {
        "name": "degree_assortativity",
        # "alias": "Asortatividad de grado"
        "alias": "Degree assortativity"
    },
    {
        "name": "hindex_assortativity",
        # "alias": "Asortatividad de índice h"
        "alias": "H-index assortativity"
    },
    {
        "name": "publications_assortativity",
        # "alias": "Asortatividad de publicaciones"
        "alias": "Publications assortativity"
    },
    {
        "name": "cites_assortativity",
        # "alias": "Asortatividad de citas"
        "alias": "Cites assortativity"
    },
    
    # Researcher info (basic metrics):
    # 'cvu',
    # 'names', 'surnames', 'nobilis', 'area',

    # Researcher productivity info:
    {
        "name": "hindex",
        # "alias": "Índice h"
        "alias": "H-index"
    },
    {
        "name": "cites_count",
        # "alias": "Número de citas"
        "alias": "Cites count"
    },
    {
        "name": "publications_count",
        # "alias": "Número de publicaciones"
        "alias": "Publications count"
    },
    
    # Career length related info:
    {
        "name": "init_year",
        # "alias": "Año de inicio"
        "alias": "Start year"
    },
    {
        "name": "career_length",
        # "alias": "Duración de carrera"
        "alias": "Career length"
    }
    # 'category',

    # Discipline related:
    # 'subdiscipline', 'speciality',

    # Affiliation related:
    # 'country', 'state', 'federal_entity'
]


# names of predictors that should be processed by pandas get_dummies() method
PREDICTORS_DUMMIES_COLUMN_NAMES = ["area","category",'subdiscipline', 'speciality','country', 'state', 'federal_entity']

CSV_FILES_INFO = {
    "salud":{
        "csv_path": HEALTH_CSV_FILENAME
    },
    "economistas":{
        "csv_path": ECONOMISTS_CSV_FILENAME
    },
    "ingenieros":{
        "csv_path": ENGINEERS_CSV_FILENAME
    }
}

# for comparison report
DUMMY_STRATEGIES_INFO = {
    "stratified": {
        # "alias": "Estratificado"
        "alias": "Stratified"
    },
    "most_frequent": {
        # "alias": "Más frecuente"
        "alias": "Most frequent"
    },
    "uniform": {
        # "alias": "Uniforme"
        "alias": "Uniform"
    }
}

COLUMN_NAME_RANDOM_FOREST = "Random forest" # for comparison report
COLUMN_NAME_TARGET = "Target category" #"Categoría Objetivo" # for comparison report

##################################################################################################################
#                                    DERIVED CONSTANTS FROM CONFIGURATION
##################################################################################################################
PREDICTORS_MAP= {prec_inf["name"]:prec_inf for prec_inf in PREDICTORS}
PREDICTORS_COLUMN_NAMES = [pred_info["name"] for pred_info in PREDICTORS]



##################################################################################################################
#                                                  FUNCTION DEFINITIONS
##################################################################################################################

def last_minute_fix_dataframe(
    aff_cat_col_name:str,
    data:Any
):
    
    # LAST MINUTE FIX:
    data.loc[data[aff_cat_col_name] == 'NORESTE', aff_cat_col_name] = 'NORTE'
    data.loc[data[aff_cat_col_name] == 'NOROESTE', aff_cat_col_name] = 'NORTE'
    data.loc[data[aff_cat_col_name] == 'SURESTE', aff_cat_col_name] = 'SUR'
    
    # OPCION 1
    # Categoría 1. Federal + institutos
    # Categoría 2. Todo lo demas
    # data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = 'CAT1'
    # data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = 'CAT1'
    #
    # data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = 'CAT2'
    # data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = 'CAT2'
    # data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = 'CAT2'
    # data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = 'CAT2'
    # data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = 'CAT2'

    # OPCION 2
    #  Primero se cambia NORESTE -> NORTE, NOROESTE->NORTE y SURESTE->SUR
    #  Luego se definen categorias: 
    # Cat. 1 FEDERAL
    # Cat. 2 INSTITUTO
    # Cat. 3 OCCIDENTE,NORTE,CENTRO, SUR
    # Cat. 4 SIN_INST
    data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = CAT_1_NAME

    data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = CAT_2_NAME

    data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = CAT_3_NAME
    data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = CAT_3_NAME
    data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = CAT_3_NAME    
    data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = CAT_3_NAME
    
    data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = CAT_4_NAME

    # OPCION 3
    #1. Federales + institutos
    #2. Sin sin instituciones + sur
    #3. Todo los demás
    # data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = 'CAT1'
    # data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = 'CAT1'

    # data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = 'CAT2'
    # data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = 'CAT2'

    # data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = 'CAT3'
    # data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = 'CAT3'
    # data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = 'CAT3'
    
    # OPCION 4
    # 1. Instituto + federal
    # 2. Occidente 
    # 3. Norte
    # 4. Centro
    # 5. Sur + sin_inst
    # data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = 'CAT1'
    # data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = 'CAT1'

    # data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = 'CAT2'

    # data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = 'CAT3'

    # data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = 'CAT4'

    # data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = 'CAT5'
    # data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = 'CAT5'

    # OPCION 5
    # 1. Federal
    # 2. Instituto 
    # 3. Norte + occidente 
    # 4. Centro
    # 5. Sur+ sin inst
    # data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = 'CAT1'

    # data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = 'CAT2'
    
    # data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = 'CAT3'
    # data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = 'CAT2'

    # data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = 'CAT4'

    # data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = 'CAT5'
    # data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = 'CAT5'
    
def get_short_name_from_csv_path(csv_path:str)->str:
    for alias, csv_info in CSV_FILES_INFO.items():
        if csv_info["csv_path"] == csv_path:
            return alias
    raise Exception("FATAL ERROR, can't find alias for csv path:" + csv_path)

def make_prediction(
    csv_files:list[str],
    predictors_names:list[str],
    predictors_dummies_names:list[str],
    target:str,
    importance_threshold:float,
    random_state:int     
)->dict[str,Any]:    

    data = rfc.build_data(
        csv_files = csv_files,        
        column_names = predictors_names + ALL_TARGETS,
        remove_rows_with_nan=True
    )
    
    last_minute_fix_dataframe(
        data=data,
        aff_cat_col_name=TARGET_AFFILIATION
    )

    predictors,target,predictors_names,index_val_dic, val_index_dic = rfc.extract_predictors_and_target(
        data=data,
        predictors_names=predictors_names,
        predictors_dummies_names=predictors_dummies_names,
        target_name=target
    )
    train_predictors, test_predictors, train_target, test_target = rfc.build_training_sets(
                                                                        predictors=predictors,
                                                                        target=target,
                                                                        random_state=random_state       
                                                                    )
    random_forest_model = rfc.train_random_forest_classifier(
        train_predictors=train_predictors,
        train_target=train_target,
        random_state=random_state
    )

    # Use the trained model using test data
    print("Making predictions")
    predictions = random_forest_model.predict(test_predictors)
    # get target and predictions formatted
    train_target_fmt = rfc.map_values_to_names(values=train_target,val_names_val_dic=index_val_dic)
    test_target_fmt = rfc.map_values_to_names(values=test_target,val_names_val_dic=index_val_dic)
    predictions_fmt = rfc.map_values_to_names(values=predictions,val_names_val_dic=index_val_dic)
    
    conf_matrix_labels = list(set(test_target_fmt+predictions_fmt))
    conf_matrix = confusion_matrix(y_true=test_target_fmt,y_pred=predictions_fmt,labels=conf_matrix_labels)
    class_rep = classification_report(y_true=test_target_fmt,y_pred=predictions_fmt,output_dict=True)

    print('Confusion Matrix:\n ', conf_matrix,'\n')
    print(classification_report(y_true=test_target_fmt,y_pred=predictions_fmt,labels=conf_matrix_labels),'\n')

    predictors_importances = rfc.get_predictors_importances_by_threshold(
        predictor_importances=random_forest_model.feature_importances_,
        predictors_names=predictors_names,
        importance_threshold=importance_threshold
    )

    return {
        "predictors_importances":predictors_importances,
        "train_predictors":train_predictors,
        "train_target":train_target,
        "test_predictors":test_predictors,        
        "test_target":test_target,
        # formatted vectors
        "predictions_fmt":predictions_fmt,
        "train_target_fmt":train_target_fmt,
        "test_target_fmt":test_target_fmt,                
        # dataframe data
        "data":data,
        # predictions report info
        "confusion_matrix":conf_matrix,
        "confusion_matrix_labels":conf_matrix_labels,
        "classification_report":class_rep        
    }

def print_accuracy_generic_report()->None:
    report = dict()
    for csv_files_tuple in mu.all_subsets(CSV_FILE_PATHS,exclude_empty_set=True):
        csv_files = list(csv_files_tuple)        
        print(csv_files)
        result = make_prediction(
            csv_files=csv_files,
            predictors_names=PREDICTORS_COLUMN_NAMES,
            predictors_dummies_names=PREDICTORS_DUMMIES_COLUMN_NAMES,
            target=TARGET,
            importance_threshold=0.8, # get predictors contributing 80% to total importance
            random_state=RANDOM_STATE
        )
        accuracy = accuracy_score(result["test_target_fmt"],result["predictions_fmt"])
        report[csv_files_tuple]= accuracy
    pprint(report)    


def print_accuracy_after_threshold()->None:
    
    result = make_prediction(
        csv_files=CSV_FILE_PATHS,
        predictors_names=PREDICTORS_COLUMN_NAMES,
        predictors_dummies_names=PREDICTORS_DUMMIES_COLUMN_NAMES,
        target=TARGET,
        importance_threshold=0.8, # get predictors contributing 80% to total importance
        random_state=RANDOM_STATE
    )
    #exit()
    new_predictors = [tuple_info[0] for tuple_info in result["predictors_importances"]]
    new_result = make_prediction(
        csv_files=CSV_FILE_PATHS,
        predictors_names=new_predictors,
        predictors_dummies_names=PREDICTORS_DUMMIES_COLUMN_NAMES,
        target=TARGET,
        importance_threshold=1,
        random_state=RANDOM_STATE
    )
    
    
def config_plot(filename:str, g):    
    for key,config in cppr.IMG_CONFIG_PROPERTIES_EN.items():
        parts = key.split("/")
        csv_files = parts[1]
        target = parts[2]
        predictor = parts[3]
        if csv_files in filename and target in filename and predictor in filename:
            ax = g.axes[0, 0]  # loop if multiple axes
            x0 = config[cppr.PROP_X_LIM][0]
            x1 = config[cppr.PROP_X_LIM][1]
            if x0 is not None:
                ax.viewLim.x0=x0
            if x1 is not None:
                ax.viewLim.x1=x1
            break
        
        
def save_kde(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:
    # KDE
    file_path = f"{file_path_preffix}-kde.{file_extension}"          
    #g = sns.displot(data=data, x=main_column, hue=secondary_column, stat="probability")
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, kind="kde",palette=PALETA_COLORES)
    #g = sns.displot(data=data, x=main_column, hue=secondary_column,kind="kde",fill=True)
    #g = sns.displot(data=data, x=main_column, hue=secondary_column, stat="probability",kde=True)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    plt.savefig(file_path,format=file_extension)
    plt.close()
    
    
def save_cumulative(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:
    #  CUMULATIVE
    file_path = f"{file_path_preffix}-cum.{file_extension}"      
    #g = sns.displot(data=data, x=main_column, hue=secondary_column, stat="probability")
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, kind="ecdf",palette=PALETA_COLORES)
    #g = sns.displot(data=data, x=main_column, hue=secondary_column,kind="kde",fill=True)
    #g = sns.displot(data=data, x=main_column, hue=secondary_column, stat="probability",kde=True)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path,format=file_extension)
    plt.close()


def save_hist(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:
    # HIST PROBABILITY 
    file_path = f"{file_path_preffix}-hist.{file_extension}"      
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, stat="probability",palette=PALETA_COLORES)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path,format=file_extension)
    plt.close()
    
    
def save_hist_step_fill(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:   
    # HIST PROBABILITY STEP FILL
    file_path = f"{file_path_preffix}-hist_step_fill.{file_extension}"      
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, stat="probability", element="step",palette=PALETA_COLORES)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path,format=file_extension)
    plt.close()
    
    
def save_hist_step_nofill(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:   
    # HIST PROBABILITY STEP NO FILL
    file_path = f"{file_path_preffix}-hist_step_nofill.{file_extension}"      
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, stat="probability", element="step",fill=False,palette=PALETA_COLORES)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path,format=file_extension)
    plt.close()
        
    
def save_hist_stack(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:   
    # HIST PROBABILITY STACK
    file_path = f"{file_path_preffix}-hist_stack.{file_extension}"      
    g = sns.displot(data=data,
                    x=predictor_name,
                    hue=target_name,
                    hue_order=target_values,
                    stat="probability",
                    multiple="stack",
                    palette=PALETA_COLORES
        )    
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path)
    plt.close()


def save_hist_dodge(
    file_path_preffix:str,
    file_extension:str,
    predictor_name:str,
    target_name:Optional[str],
    target_values:list[str],
    y_label:str,
    data:Any,        
)->None:     
    # HIST PROBABILITY DODGE
    file_path = f"{file_path_preffix}-hist_dodge.{file_extension}"      
    g = sns.displot(data=data, x=predictor_name, hue=target_name, hue_order=target_values, stat="probability", multiple="dodge",palette=PALETA_COLORES)
    g.set_axis_labels(x_var=predictor_name, y_var=y_label)
    config_plot(file_path,g)    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    #plt.show()
    plt.savefig(file_path,format=file_extension)
    plt.close()


def save_distribution_image_file(
    data:Any,
    predictor_name:str,
    y_label:str,
    file_path_preffix:str,    
    file_extension:str,
    target_name:Optional[str]=None    
)->None:    
    sns.set_theme(style="whitegrid")
    target_values = data[target_name].unique()
    target_values.sort()
    
    # save_kde(
    #     file_path_preffix=file_path_preffix,
    #     file_extension=file_extension,
    #     predictor_name=predictor_name,
    #     target_name=target_name,
    #     target_values=target_values,
    #     y_label=y_label,
    #     data=data        
    # )
    
    # save_cumulative(
    #     file_path_preffix=file_path_preffix,
    #     file_extension=file_extension,
    #     predictor_name=predictor_name,
    #     target_name=target_name,
    #     target_values=target_values,
    #     y_label=y_label,
    #     data=data        
    # )
    
    # save_hist(
    #    file_path_preffix=file_path_preffix,
    #    file_extension=file_extension,
    #    predictor_name=predictor_name,
    #    target_name=target_name,
    #    target_values=target_values,
    #    y_label=y_label,
    #    data=data        
    #)
    
    # save_hist_step_fill(
    #     file_path_preffix=file_path_preffix,
    #     file_extension=file_extension,
    #     predictor_name=predictor_name,
    #     target_name=target_name,
    #     target_values=target_values,
    #     y_label=y_label,
    #     data=data        
    # )
    
    # save_hist_step_nofill(
    #     file_path_preffix=file_path_preffix,
    #     file_extension=file_extension,
    #     predictor_name=predictor_name,
    #     target_name=target_name,
    #     target_values=target_values,
    #     y_label=y_label,
    #     data=data        
    # )
    
    save_hist_stack(
        file_path_preffix=file_path_preffix,
        file_extension=file_extension,
        predictor_name=predictor_name,
        target_name=target_name,
        target_values=target_values,
        y_label=y_label,
        data=data        
    )
    
    # save_hist_dodge(
    #     file_path_preffix=file_path_preffix,
    #     file_extension=file_extension,
    #     predictor_name=predictor_name,
    #     target_name=target_name,
    #     target_values=target_values,
    #     y_label=y_label,
    #     data=data        
    # )
    
    
    
def save_predictors_importances(
    csv_path:str,
    predictors_importances:list[tuple[str,float]],
)->None:
    #rows:list[Any] = [["Orden","Predictor", "Importancia"]]
    rows:list[Any] = [["Order","Predictor", "Importance"]]
    for index,predictor_info in enumerate(predictors_importances):
        order = index + 1
        predictor_name = predictor_info[0]        
        predictor_importance = predictor_info[1]
        predictor_alias = PREDICTORS_MAP[predictor_name]["alias"]
        rows.append([order, predictor_alias,predictor_importance])
        
    fu.save_list_as_csv(filename=csv_path,data=rows)
    
def save_classification_report(
    csv_path:str,
    classification_report:Any
)->None:        
    # Convert the report to a DataFrame
    df = pd.DataFrame(classification_report).transpose()
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path)
    
def save_confussion_matrix(
    csv_path:str,
    confussion_matrix:Any,
    labels:list[str]
)->None:    

    # Convert the confusion matrix to a DataFrame
    index = [f'Actual {name}' for name in labels]
    #columns = [f'Predicción {name}' for name in labels]
    columns = [f'Prediction {name}' for name in labels]
    df_cm = pd.DataFrame(confussion_matrix, index=index,columns=columns)

    # Save the DataFrame to a CSV file
    df_cm.to_csv(csv_path)

def get_dummy_classifier_comparison_dict(
    target_alias:str,
    X_train:Any,
    y_train:Any,
    y_test:Any,
    X_test:Any,
    predictions:Any
)->tuple[dict[str,Any],dict[str,Any]]:
    
    rfc_accuracy = accuracy_score(y_test,predictions).round(decimals=2)
    comparison = {
        COLUMN_NAME_RANDOM_FOREST:rfc_accuracy,
        COLUMN_NAME_TARGET:target_alias
    }
    
    comparison_percentages:dict[str,Any] = {
        COLUMN_NAME_TARGET:target_alias
    }
    
    for dummy_strategy, dummy_strategy_info in DUMMY_STRATEGIES_INFO.items():
        #Literal["stratified","most_frequent","uniform"]
        dummy_strategy_alias = dummy_strategy_info["alias"]
        dummy_clf = DummyClassifier(strategy=dummy_strategy,random_state=0)
        dummy_clf.fit(X_train, y_train) 
        dummy_accuracy = accuracy_score(y_test,dummy_clf.predict(X_test)).round(decimals=2)
        # add to dict
        comparison[dummy_strategy_alias]=dummy_accuracy
        # i.e. 100% means twice as good
        percentage = (rfc_accuracy-dummy_accuracy)*100/dummy_accuracy
        comparison_percentages[dummy_strategy_alias]= percentage.round(decimals=2)
        
    return comparison,comparison_percentages




def save_predictors_report(
    report_dir:str,    
    target_name:str,
    predictions_result:Any    
)->Any:
    target_alias = VAL_ALIAS_MAPS[target_name]["alias"]
    data = predictions_result["data"]            
    predictors_importances = predictions_result["predictors_importances"]    
    class_rep = predictions_result["classification_report"]
    conf_mat = predictions_result["confusion_matrix"]
    conf_mat_labels = predictions_result["confusion_matrix_labels"]
    test_target_fmt = predictions_result["test_target_fmt"]
    predictions_fmt = predictions_result["predictions_fmt"]                  
    test_predictors = predictions_result["test_predictors"]
    train_predictors = predictions_result["train_predictors"]
    train_target_fmt = predictions_result["train_target_fmt"]
        
    #exit()
    new_predictors = [tuple_info[0] for tuple_info in predictors_importances]
    for index,predictor_name in enumerate(new_predictors):
        predictor_alias = PREDICTORS_MAP[predictor_name]["alias"]
        plot_data = data[[predictor_name,target_name]]
        val_alias_map = VAL_ALIAS_MAPS[target_name]["val_alias_map"]
        # rename values in target:
        for val,name in val_alias_map.items():
            plot_data.loc[plot_data[target_name] == val, target_name] = name
        # rename columns
        plot_data = plot_data.rename(columns={predictor_name:predictor_alias,target_name:target_alias})    
        
        #file_path_preffix = f"{report_dir}/{target_alias}-{index+1}_{predictor_alias}"
        file_path_preffix = f"{report_dir}/{index+1}_{predictor_alias}"
        file_path_preffix = file_path_preffix.replace(" ","_")
        save_distribution_image_file(
            file_path_preffix=file_path_preffix,
            data=plot_data,
            predictor_name=predictor_alias,
            target_name=target_alias,
            y_label=Y_LABEL,
            file_extension=FIGURE_FORMAT
        )
        
    csv_path = f"{report_dir}/{target_alias}-importancia_predictores.csv"
    save_predictors_importances(
        csv_path=csv_path,
        predictors_importances=predictors_importances
    )
        
    csv_path = f"{report_dir}/{target_alias}-classification_report.csv"
    save_classification_report(
        csv_path=csv_path,
        classification_report=class_rep
    )
    
    csv_path = f"{report_dir}/{target_alias}-confussion_matrix.csv"
    save_confussion_matrix(
        csv_path=csv_path,
        confussion_matrix=conf_mat,
        labels=conf_mat_labels
    )
    
    csv_path = f"{report_dir}/{target_alias}-dummy_comparison.csv"
    comparison,comparison_percentages = get_dummy_classifier_comparison_dict(
        target_alias=target_alias,
        X_train=train_predictors,
        y_train=train_target_fmt,
        y_test=test_target_fmt,
        X_test=test_predictors,
        predictions=predictions_fmt
    )
    
    return {
        "comparison":comparison,
        "comparison_percentages":comparison_percentages
    }
        

def save_reports()->None:
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    general_comparison_report = []
    general_comparison_percentages_report = []
    comparison_column_names:list[str] = []
    comparison_percentages_column_names:list[str] = []
    for csv_files in mu.all_subsets(CSV_FILE_PATHS,exclude_empty_set=True):
        if len(csv_files) != 3:
            continue
        # build folder to store all predictors data
        csv_short_names = [get_short_name_from_csv_path(csv_path) for csv_path in csv_files]
        folder_name = '_'.join(csv_short_names)
        comparison_report = []
        comparison_report_percentages = []
        for target_name in ALL_TARGETS:    
            # skip if target=discipline and only one csv file is used (this type of prediction is useless)
            if target_name == TARGET_DISCIPLINE and len(csv_files)==1:                
                continue
            
            # make and get predictions info
            result = make_prediction(
                csv_files=csv_files,
                predictors_names=PREDICTORS_COLUMN_NAMES,
                predictors_dummies_names=PREDICTORS_DUMMIES_COLUMN_NAMES,
                target=target_name,
                importance_threshold=IMPORTANCE_THRESHOLD, # get predictors contributing 80% to total importance
                random_state=RANDOM_STATE        
            )       
            # make predictions again using most important predictors
            new_predictors = [tuple_info[0] for tuple_info in result["predictors_importances"]]
            result = make_prediction(
                csv_files=csv_files,
                predictors_names=new_predictors,
                predictors_dummies_names=PREDICTORS_DUMMIES_COLUMN_NAMES,
                target=target_name,
                importance_threshold=1,
                random_state=RANDOM_STATE
            )                 
            
            target_alias = VAL_ALIAS_MAPS[target_name]["alias"]
            report_dir = f"{REPORT_DIR}/{folder_name}/{target_alias}"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            # save info    
            reports = save_predictors_report(
                report_dir=report_dir,                
                target_name=target_name,
                predictions_result=result
            )
            
            comparison = reports["comparison"]
            comparison_percentages = reports["comparison_percentages"]
            comparison_report.append(comparison)
            comparison_report_percentages.append(comparison_percentages)
            if len(comparison_column_names) == 0:
                comparison_column_names = list(comparison.keys())
            if len(comparison_percentages_column_names) == 0:
                comparison_percentages_column_names = list(comparison_percentages.keys())
                
                
        csv_path = f"{REPORT_DIR}/{folder_name}/dummy_classifiers-comparison.csv"
        fu.dictionary_lists_to_csv(
            csv_filepath=csv_path,
            rows=comparison_report
        )
        temp_val = ",".join(csv_short_names)
        temp_val = f"Exactidud de los siguientes objetivos: {temp_val}"
        temp_header = {col_name:"" for col_name in comparison_column_names}
        temp_header[COLUMN_NAME_TARGET] = temp_val
        comparison_report.insert(0,temp_header)
        general_comparison_report.extend(comparison_report)
                
                
        csv_path = f"{REPORT_DIR}/{folder_name}/dummy_classifiers-comparison-percentages.csv"
        fu.dictionary_lists_to_csv(
            csv_filepath=csv_path,
            rows=comparison_report_percentages
        )
        temp_val = ",".join(csv_short_names)
        temp_val = f"Exactidud de los siguientes objetivos: {temp_val}"
        temp_header = {col_name:"" for col_name in comparison_percentages_column_names}
        temp_header[COLUMN_NAME_TARGET] = temp_val
        comparison_report_percentages.insert(0,temp_header)
        general_comparison_percentages_report.extend(comparison_report_percentages)
    
    # Save general reports
    csv_path = f"{REPORT_DIR}/general-dummy_classifiers-comparison.csv"
    fu.dictionary_lists_to_csv(
        csv_filepath=csv_path,
        rows=general_comparison_report
    )
    
    csv_path = f"{REPORT_DIR}/general-dummy_classifiers-comparison-percentages.csv"
    fu.dictionary_lists_to_csv(
        csv_filepath=csv_path,
        rows=general_comparison_percentages_report
    )


##################################################################################################################
#                                                  RUNNING!!
##################################################################################################################            
#print_accuracy_generic_report()
#print_accuracy_after_threshold()

save_reports()

