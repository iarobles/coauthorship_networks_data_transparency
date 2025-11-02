import os
from typing import Any, Optional
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#############################################################################################################
#  CONFIGURATION VARIABLES
#############################################################################################################
CSV_PATH_SNII = "./data/snii_data/original/all_persons_health_economists_engineering.csv"

FIG_FORMAT = "png" # eps, png, pdf or jpg
COLOR_PALETTE = ['#EB6123','#512888','#dc267f','#648fff']

REPORTS_FOLDER = "./reports/paper_general_graphs"
IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE = REPORTS_FOLDER + "/discipline_gender"
IMAGE_FILE_PREFIX_AFFILIATION_BY_DISCIPLINE = REPORTS_FOLDER + "/discipline_affiliation"
IMAGE_FILE_PREFIX_PAPERS_BY_DISCIPLINE = REPORTS_FOLDER + "/discipline_papers"

COL_CVU = "CVU"  # "CVU"
COL_DISCIPLINE = "Discipline"  # "Discipline"
COL_AFFILIATION = 'Affiliation'  # 'Affiliation'
COL_GENDER = 'Gender'  # 'Gender'
COL_COMPLETE = 'Complete'  # 'Complete'
# COLUMNS BUILD 
COL_TOTAL = 'Total'  # 'Total'
FULL_GENDER_CONCAT_COLUMN = "Complete - Gender"  # "Complete - Gender"

# Map to change csv column names to new names
RENAME_COLS_MAP = {
    'cvu': COL_CVU,
    'disciplina': COL_DISCIPLINE,    
    # The original affiliation from the beginning is only
    # for internal control (ignore):
    'afiliacion': 'original_afiliacion',     
    # For internal control, ignore:
    'completo': 'original_completo',        
    # The affiliation according to the 4 categories 
    'categoria_afiliacion': COL_AFFILIATION,
    'genero': COL_GENDER,    
    # Whether or not their data is complete
    'metrica_calculada':  COL_COMPLETE
}

# map to rename certain values for given columns 
# (e.g., for Gender "f" is set to "Female")
VALUE_COMPLETE = "Complete"  # "Complete"
VALUE_INCOMPLETE = "Incomplete"  # "Incomplete"
TITLE_COMPLETE = VALUE_COMPLETE  # "Complete"
TITLE_INCOMPLETE = VALUE_INCOMPLETE  # "Incomplete"
TITLE_COMPLETE_AND_INCOMPLE = "Complete and Incomplete"  # "Complete and Incomplete"
X_LABEL_DISCIPLINE = "Discipline"  # "Discipline"
Y_LABEL_TOTAL_PEOPLE = "Total people"  # "Total people"

NAME_ENGINEERING = "Engineering"  # "Engineering"      
NAME_HEALTH_SCIENCES = "Health Sci."  # "Health Sci."
NAME_ECONOMIC_SCIENCES = "Economic Sci."  # "Economic Sci."

VALUE_RENAME_MAP = {
    COL_DISCIPLINE:{
        "INGENIERIA": NAME_ENGINEERING,
        "CIENCIAS DE LA SALUD": NAME_HEALTH_SCIENCES,
        "CIENCIAS ECONOMICAS": NAME_ECONOMIC_SCIENCES
    },
    COL_GENDER:{        
        "f": "Female",#"Female",
        "m": "Male"#"Male"
    },
    COL_COMPLETE:{
        True: VALUE_COMPLETE,
        False:VALUE_INCOMPLETE
    },
    COL_AFFILIATION:{
        "Categoría 1":"Category 1",
        "Categoría 2":"Category 2",
        "Categoría 3":"Category 3",
        "Categoría 4":"Category 4"
    }
}


COL_TOTAL_PUBLICATIONS = "Total publications"  # Total publications
COL_PUBLICATIONS = "Publications"  # Publications
COL_AUTHORS_WITH_SNII_PUBLICATIONS = "From authors with SNII"  # From authors with SNII
COL_COAUTHORS_WITHOUT_SNII_PUBLICATIONS = "From coauthors without SNII"  # From coauthors without SNII
Y_LABEL_TOTAL_PUBLICATIONS = "Total publications"  # Total publications

TOTAL_PAPERS_PER_AUTHOR_AND_COAUTHOR = {
    COL_DISCIPLINE: [
                        NAME_HEALTH_SCIENCES,                        
                        NAME_ECONOMIC_SCIENCES, 
                        NAME_ENGINEERING,
                        NAME_HEALTH_SCIENCES,                        
                        NAME_ECONOMIC_SCIENCES, 
                        NAME_ENGINEERING,
                    ],
    COL_TOTAL_PUBLICATIONS: [
                        2210153, # health coauthors
                        212468,  # economics coauthors
                        376146,  # engineering coauthors
                        23861,   # health snii authors
                        7116,    # economics snii authors
                        6799     # engineering snii authors
                       ],
    COL_PUBLICATIONS: [COL_COAUTHORS_WITHOUT_SNII_PUBLICATIONS,
                        COL_COAUTHORS_WITHOUT_SNII_PUBLICATIONS, 
                        COL_COAUTHORS_WITHOUT_SNII_PUBLICATIONS,
                        COL_AUTHORS_WITH_SNII_PUBLICATIONS, 
                        COL_AUTHORS_WITH_SNII_PUBLICATIONS,
                        COL_AUTHORS_WITH_SNII_PUBLICATIONS]
}


#############################################################################################################
#  FUNCTION DEFINITIONS
#############################################################################################################

def build_dataframe():
    df = pd.read_csv(CSV_PATH_SNII)
    # Renames the column names to more appropriate names for the report
    df = df.rename(columns=RENAME_COLS_MAP)
    # Renames values in some columns
    for col_name, value_map in VALUE_RENAME_MAP.items():
        df[col_name] = df[col_name].map(
            value_map,
            na_action='ignore'
        )
    # Add a new column
    df[FULL_GENDER_CONCAT_COLUMN] = df[COL_COMPLETE] + ' - ' + df[COL_GENDER]
    return df

def group_dataframe_by_columns_fill_with_zero(
    df:Any,
    columns:list[str],
    col_name_total:str,
    fill_missing_with_zero:bool
):
    
    if fill_missing_with_zero is False:
        aggregated = df.groupby(columns).size().reset_index(
            name=col_name_total
        ).sort_values(
           by=columns
        )
        return aggregated
    
    # group
    aggregated = df.groupby(columns).size().reset_index(name=col_name_total)

    # Create multi-index with all possible combinations
    unique_values = []
    for col in columns:
        unicos = df[col].unique()
        unique_values.append(unicos)
        
    multi_index = pd.MultiIndex.from_product(
        unique_values,
        names=columns
    )

    # reindex and fill values not in the multi-index with 0
    aggregated = aggregated.set_index(columns).reindex(multi_index, fill_value=0).reset_index()

    # sort the result
    aggregated = aggregated.sort_values(by=columns)
    
    return aggregated


def add_totals_to_distplot_stack(
    g:Any,
    total_hue:int
):
    # Access the axes from the FacetGrid
    axes = g.axes.flatten()
    total_p = len(axes[0].patches)
    total_bars = int(total_p/total_hue)
    p_last_bar_index = total_p - total_bars
    bar_heights = [0 for _ in range(0,total_bars)]
    for ax in axes:                   
        for p_index, p in enumerate(ax.patches):
            bar_index = p_index % total_bars
            height = bar_heights[bar_index-1] = bar_heights[bar_index-1] + p.get_height()
            # set the total value
            if p_index>=p_last_bar_index:
                x = p.get_x() + p.get_width() / 2
                ax.annotate(f'{int(height)}',
                            xy=(x, height),
                            xytext=(0, 3),  # Vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                
def add_totals_to_catplot(
    g:Any
):
    total = len(g.axes[0])
    # Place the number of people for each bar
    for i in range(0,total):
        ax = g.axes[0,i]
        for i in ax.containers:
            ax.bar_label(i,fmt='%d')    
    
    
def stacked_plot(
    df:Any,
    file_path:str,
    x_col:str,
    hue_col:str,
    x_label:str,
    y_label:str,
    title:Optional[str]
):      
    g=sns.displot(df, x=x_col, hue=hue_col, multiple="stack",palette=COLOR_PALETTE)
    g.set_axis_labels(x_label, y_label)  
    total_hue = len(df[hue_col].unique())
    add_totals_to_distplot_stack(g=g,total_hue=total_hue)       
    # show borders: top, right, bottom and left 
    sns.despine(top = False, right=False, bottom=False, left=False)    
    #plt.show()
    # save image
    if title is not None:
        plt.title(title)
    plt.savefig(file_path)
    plt.close()
    
def bar_plot(
    df:Any,
    file_path:str,
    x_col:str,
    y_col:str,    
    x_label:str,
    y_label:str,
    titles:list[str],
    hue_col:Optional[str],
    col_col:Optional[str],
    color:Optional[tuple[float,float,float]]=None    
):    
    # build graph
    sns.set_palette(COLOR_PALETTE)
    g = sns.catplot(
        data=df,
        kind="bar",
        x=x_col,
        y=y_col,
        hue=hue_col,
        col=col_col,
        color=color
        #height=4.6,  # Height of each facet
        #aspect=1.5  # Width/height ratio (increase to make it wider)
        #errorbar="sd"        
    )    
    # X and Y axis labels
    g.set_axis_labels(x_label, y_label)     
    #g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    for index, title in enumerate(titles):
        # Figure name 0 (left) and 1 (right)
        g.axes[0,index].set_title(title)
    
    # Place the number of people for each bar
    add_totals_to_catplot(g) 

    # show borders: top, right, bottom and left 
    sns.despine(top = False, right=False, bottom=False, left=False)    
    #plt.show()
    # save image
    plt.savefig(file_path)
    plt.close()

def bar_plot_by_group(    
    df:Any,
    file_path:str,
    columns:list[str],
    col_name_total:str,
    fill_missing_with_zero:bool,
    x_col:str,
    y_col:str,
    hue_col:str,
    col_col:Optional[str],
    x_label:str,
    y_label:str,
    titles:list[str]
):    
    # build dataframe with aggregated data
    df = group_dataframe_by_columns_fill_with_zero(
        df=df,
        columns=columns,
        col_name_total=col_name_total,
        fill_missing_with_zero=fill_missing_with_zero
    )        
    # build graph
    bar_plot(
        df=df,
        file_path=file_path,
        x_col=x_col,
        y_col=y_col,
        hue_col=hue_col,
        col_col=col_col,
        x_label=x_label,
        y_label=y_label,
        titles=titles
    )
    

def discipline_gender_bar_plot_comp_inc(
    df:Any,
    file_path:str
):    
    bar_plot_by_group(
        df = df,
        file_path = file_path,
        columns = [COL_DISCIPLINE,COL_COMPLETE,COL_GENDER],
        col_name_total=COL_TOTAL,
        fill_missing_with_zero=True,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL,
        hue_col=COL_GENDER,
        col_col=COL_COMPLETE,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        titles=[TITLE_INCOMPLETE,TITLE_COMPLETE]
    )
    
def discipline_gender_bar_plot_comp(
    df:Any,
    file_path:str
):    
    df = df[df[COL_COMPLETE]==VALUE_COMPLETE]
    bar_plot_by_group(
        df = df,
        file_path = file_path,
        columns = [COL_DISCIPLINE,COL_GENDER],
        col_name_total=COL_TOTAL,
        fill_missing_with_zero=True,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL,
        hue_col=COL_GENDER,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        titles=[TITLE_COMPLETE]
    )
    
    
def discipline_gender_bar_plot_inc(
    df:Any,
    file_path:str
):    
    df = df[df[COL_COMPLETE]==VALUE_INCOMPLETE]
    bar_plot_by_group(
        df = df,
        file_path = file_path,
        columns = [COL_DISCIPLINE,COL_GENDER],
        col_name_total=COL_TOTAL,
        fill_missing_with_zero=True,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL,
        hue_col=COL_GENDER,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        titles=[TITLE_INCOMPLETE]
    )
    
def discipline_gender_stack_plot_comp_inc(
    df:Any,
    file_path:str
):
    stacked_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        hue_col=FULL_GENDER_CONCAT_COLUMN,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        title=TITLE_COMPLETE_AND_INCOMPLE
    )                
    
def discipline_gender_stack_plot_comp(
    df:Any,
    file_path:str
):
    df = df[df[COL_COMPLETE]==VALUE_COMPLETE]
    stacked_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        hue_col=COL_GENDER,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        title=TITLE_COMPLETE
    )                

def discipline_gender_stack_plot_inc(
    df:Any,
    file_path:str
):
    df = df[df[COL_COMPLETE]==VALUE_INCOMPLETE]
    stacked_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        hue_col=COL_GENDER,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        title=TITLE_INCOMPLETE
    )                
        
    
def discipline_affiliation_bar_plot_comp(
    df:Any,
    file_path:str
):  
    df = df[df[COL_COMPLETE]==VALUE_COMPLETE]
    # build dataframe with aggregated data
    df = group_dataframe_by_columns_fill_with_zero(
        df=df,
        columns=[COL_DISCIPLINE,COL_AFFILIATION],
        col_name_total=COL_TOTAL,
        fill_missing_with_zero=True
    )    
    
    bar_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL,
        hue_col=COL_AFFILIATION,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        titles=[TITLE_COMPLETE]
    )

def discipline_affiliation_stack_plot_comp(
    df:Any,
    file_path:str
):  
    df = df[df[COL_COMPLETE]==VALUE_COMPLETE]    
    stacked_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        hue_col=COL_AFFILIATION,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PEOPLE,
        title=TITLE_COMPLETE
    )                
 

def discipline_papers_bar_plot_authors_coauthors(
    df:Any,
    file_path:str
):          
    bar_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL_PUBLICATIONS,
        hue_col=COL_PUBLICATIONS,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PUBLICATIONS,
        #titles=["SNII publications and coauthors without SNII"]
        titles = ["SNII publications and coauthors without SNII"]
    )
    
def discipline_papers_bar_plot_authors(
    df:Any,
    file_path:str
):          
    df = df[df[COL_PUBLICATIONS] == COL_AUTHORS_WITH_SNII_PUBLICATIONS]
    paleta_cols = COLOR_PALETTE#sns.color_palette()

    # Identify the color for the second bar (orange in the default theme)
    orange_color = paleta_cols[1]
    bar_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL_PUBLICATIONS,
        hue_col=None,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PUBLICATIONS,
        #titles=["SNII publications"],
        titles = ["SNII publications"],
        color=orange_color
    )
    
def discipline_papers_bar_plot_coauthors(
    df:Any,
    file_path:str
):          
    df = df[df[COL_PUBLICATIONS] == COL_COAUTHORS_WITHOUT_SNII_PUBLICATIONS]
    bar_plot(
        df=df,
        file_path=file_path,
        x_col=COL_DISCIPLINE,
        y_col=COL_TOTAL_PUBLICATIONS,
        hue_col=None,
        col_col=None,
        x_label=X_LABEL_DISCIPLINE,
        y_label=Y_LABEL_TOTAL_PUBLICATIONS,
        #titles=["Co-authors without SNII publications"]
        titles = ["Co-authors without SNII publications"]
    )
      

    
def build_final_images():    
    sns.set_theme(style="whitegrid")
    if not os.path.exists(REPORTS_FOLDER):
        os.makedirs(REPORTS_FOLDER)
    df = build_dataframe()
    
    ####################################### GENDER BY DISCIPLINE ######################################
    file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-barras_comp." + FIG_FORMAT
    discipline_gender_bar_plot_comp(
        df=df,
        file_path=file_path
    )
    print("Saving gender by discipline in", file_path)
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
    
    file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-barras_inc." + FIG_FORMAT
    discipline_gender_bar_plot_inc(
        df=df,
        file_path=file_path
    )
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
        
    file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-barras_inc_comp." + FIG_FORMAT
    discipline_gender_bar_plot_comp_inc(
        df=df,
        file_path=file_path
    )
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
    # file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-stack_comp_inc." + FIG_FORMAT
    # discipline_gender_stack_plot_comp_inc(
    #     df=df,
    #     file_path=file_path
    # )
    # file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-stack_comp." + FIG_FORMAT
    # discipline_gender_stack_plot_comp(
    #     df=df,
    #     file_path=file_path
    # )
    # file_path = IMAGE_FILE_PREFIX_GENDER_BY_DISCIPLINE + "-stack_inc." + FIG_FORMAT
    # discipline_gender_stack_plot_inc(
    #     df=df,
    #     file_path=file_path
    # )
    
    ####################################### AFFILIATION BY DISCIPLINE ######################################
    file_path = IMAGE_FILE_PREFIX_AFFILIATION_BY_DISCIPLINE + "-barras_comp." + FIG_FORMAT
    discipline_affiliation_bar_plot_comp(
        df=df,
        file_path=file_path
    )    
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
        
    # file_path = IMAGE_FILE_PREFIX_AFFILIATION_BY_DISCIPLINE + "-stack_comp." + FIG_FORMAT
    # discipline_affiliation_stack_plot_comp(
    #     df=df,
    #     file_path=file_path
    # )
    # df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
    ####################################### PAPERS BY DISCIPLINE ######################################
    # Create DataFrame
    df2 = pd.DataFrame(TOTAL_PAPERS_PER_AUTHOR_AND_COAUTHOR)    
    file_path = IMAGE_FILE_PREFIX_PAPERS_BY_DISCIPLINE + "-barras_autores_coautores." + FIG_FORMAT        
    discipline_papers_bar_plot_authors_coauthors(
        df=df2,
        file_path=file_path
    )    
    df2.to_csv(file_path.replace(FIG_FORMAT,"csv"))
    ####################################### PAPERS BY DISCIPLINE ######################################
    file_path = IMAGE_FILE_PREFIX_PAPERS_BY_DISCIPLINE + "-barras_autores." + FIG_FORMAT        
    discipline_papers_bar_plot_authors(
        df=df2,
        file_path=file_path
    )    
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))
    ####################################### PAPERS BY DISCIPLINE ######################################
    file_path = IMAGE_FILE_PREFIX_PAPERS_BY_DISCIPLINE + "-barras_coautores." + FIG_FORMAT        
    discipline_papers_bar_plot_coauthors(
        df=df2,
        file_path=file_path,
    )    
    df.to_csv(file_path.replace(FIG_FORMAT,"csv"))

 

#############################################################################################################
#  READY, SET, GO!!
#############################################################################################################
build_final_images()