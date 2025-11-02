from typing import Any
import constants as c

def fix_affiliation_categories(
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
    data.loc[data[aff_cat_col_name] == 'FEDERAL', aff_cat_col_name] = c.CAT_1_NAME

    data.loc[data[aff_cat_col_name] == 'INSTITUTO', aff_cat_col_name] = c.CAT_2_NAME

    data.loc[data[aff_cat_col_name] == 'OCCIDENTE', aff_cat_col_name] = c.CAT_3_NAME
    data.loc[data[aff_cat_col_name] == 'NORTE', aff_cat_col_name] = c.CAT_3_NAME
    data.loc[data[aff_cat_col_name] == 'CENTRO', aff_cat_col_name] = c.CAT_3_NAME
    data.loc[data[aff_cat_col_name] == 'SUR', aff_cat_col_name] = c.CAT_3_NAME

    data.loc[data[aff_cat_col_name] == 'SIN_INST', aff_cat_col_name] = c.CAT_4_NAME

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


def fix_column_values(
    VAL_ALIAS_MAPS:dict[str,dict[str,Any]]
):
    for index,col_name in enumerate(new_predictors):
            predictor_alias = PREDICTORS_MAP[col_name]["alias"]
            plot_data = data[[col_name,target_name]]
            val_alias_map = VAL_ALIAS_MAPS[target_name]["val_alias_map"]
            # rename values in target:
            for val,name in val_alias_map.items():
                plot_data.loc[plot_data[target_name] == val, target_name] = name
            # rename columns
            plot_data = plot_data.rename(columns={col_name:predictor_alias,target_name:target_alias})    
        