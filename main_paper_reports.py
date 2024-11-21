import os
from typing import Any, Optional
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#############################################################################################################
#  VARIABLES DE CONFIGURACIÓN
#############################################################################################################
RUTA_SNII_CSV = "./data/snii_data/salud_economistas_ingenieros_sni_anonymized.csv"

FORMATO_FIGURA = "eps" # pdf o jpg
PALETA_COLORES = ['#EB6123','#512888','#dc267f','#648fff']

DIRECTORIO_REPORTES = "./data/reports/graficas_paper_en"
PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA = DIRECTORIO_REPORTES + "/disciplina_genero"
PREFIJO_ARCHIVO_IMAGEN_AFILIACION_POR_DISCIPLINA = DIRECTORIO_REPORTES + "/disciplina_afiliacion"
PREFIJO_ARCHIVO_IMAGEN_PAPERS_POR_DISCIPLINA = DIRECTORIO_REPORTES + "/disciplina_papers"

COL_CVU = "CVU"  # "CVU"
COL_DISCIPLINA = "Discipline"  # "Disciplina"
COL_AFILIACION = 'Affiliation'  # 'Afiliación'
COL_GENERO = 'Gender'  # 'Género'
COL_COMPLETO = 'Complete'  # 'Completo'
# COLUMNAS CONSTRUIDAS AL PROCESAR EL DATAFRAME
COL_TOTAL = 'Total'  # 'Total'
COL_CONCATENAR_COMPLETO_GENERO = "Complete - Gender"  # "Completo - Género"

# Mapa para cambiar los nombres de las olumnas del csv a nuevos nombres
MAPA_RENOMBRAR_NOMBRE_COLS = {
    'cvu': COL_CVU,
    'disciplina': COL_DISCIPLINA,    
    # La afiliación original que se tenía desde el principio es sólo
    # para control interno (ignorar):
    'afiliacion': 'original_afiliacion',     
    # Para control interno, ignorar:
    'completo': 'original_completo',        
    # La afiliación de acuerdo a las 4 categorías 
    'categoria_afiliacion': COL_AFILIACION,
    'genero': COL_GENERO,    
    # Sí se tienen sus datos completos o no
    'metrica_calculada':  COL_COMPLETO
}

# mapa para renombrar ciertos valores para columnas dadas 
# (por ejemplo para Género "f" se pone como "Femenino")
VALOR_COMPLETO = "Complete"  # "Completos"
VALOR_INCOMPLETO = "Incomplete"  # "Incompletos"
TITULO_COMPLETOS = VALOR_COMPLETO  # "Complete"
TITULO_INCOMPLETOS = VALOR_INCOMPLETO  # "Incomplete"
TITULO_COMPLETOS_E_INCOMPLETOS = "Complete and Incomplete"  # "Completos e Incompletos"
X_LABEL_DISCIPLINA = "Discipline"  # "Disciplina"
Y_LABEL_TOTAL_PERSONAS = "Total people"  # "Total personas"

NOMBRE_INGENIERIA = "Engineering"  # "Ingeniería"      
NOMBRE_CIENCIAS_SALUD = "Health Sci."  # "CC. de la salud"
NOMBRE_CIENCIAS_ECONOMICAS = "Economic Sci."  # "CC. económicas"

MAPA_RENOMBRAR_VALORES = {
    COL_DISCIPLINA:{
        "INGENIERIA": NOMBRE_INGENIERIA,
        "CIENCIAS DE LA SALUD": NOMBRE_CIENCIAS_SALUD,
        "CIENCIAS ECONOMICAS": NOMBRE_CIENCIAS_ECONOMICAS
    },
    COL_GENERO:{        
        "f": "Female",#"Femenino",
        "m": "Male"#"Masculino"
    },
    COL_COMPLETO:{
        True: VALOR_COMPLETO,
        False:VALOR_INCOMPLETO
    },
    COL_AFILIACION:{
        "Categoría 1":"Category 1",
        "Categoría 2":"Category 2",
        "Categoría 3":"Category 3",
        "Categoría 4":"Category 4"
    }
}


COL_TOTAL_PUBLICACIONES = "Total publications"  # Total publicaciones
COL_PUBLICACIONES = "Publications"  # Publicaciones
COL_PUBLICACIONES_VALOR_AUTORES_CON_SNII = "From authors with SNII"  # De autores con SNII
COL_PUBLICACIONES_VALOR_COAUTORES_SIN_SNII = "From coauthors without SNII"  # De coautores sin SNII
Y_LABEL_TOTAL_PUBLICACIONES = "Total publications"  # Total publicaciones

TOTAL_PAPERS_POR_AUTOR_Y_COAUTOR = {
    COL_DISCIPLINA: [
                        NOMBRE_CIENCIAS_SALUD,                        
                        NOMBRE_CIENCIAS_ECONOMICAS, 
                        NOMBRE_INGENIERIA,
                        NOMBRE_CIENCIAS_SALUD,                        
                        NOMBRE_CIENCIAS_ECONOMICAS, 
                        NOMBRE_INGENIERIA,
                    ],
    COL_TOTAL_PUBLICACIONES: [
                        2210153, # salud coautores
                        212468,  # economicas coautores
                        376146,  # ingeniera coautores
                        23861,   # salud autores snii
                        7116,    # economicas autores snii
                        6799     # ingenieria autores snii
                       ],
    COL_PUBLICACIONES: [COL_PUBLICACIONES_VALOR_COAUTORES_SIN_SNII,
                        COL_PUBLICACIONES_VALOR_COAUTORES_SIN_SNII, 
                        COL_PUBLICACIONES_VALOR_COAUTORES_SIN_SNII,
                        COL_PUBLICACIONES_VALOR_AUTORES_CON_SNII, 
                        COL_PUBLICACIONES_VALOR_AUTORES_CON_SNII,
                        COL_PUBLICACIONES_VALOR_AUTORES_CON_SNII]
}


#############################################################################################################
#  DEFINICION FUNCIONES
#############################################################################################################

def construir_dataframe():
    df = pd.read_csv(RUTA_SNII_CSV)
    # Renombra los nombres de las columnas a nombres más apropiados para el reporte
    df = df.rename(columns=MAPA_RENOMBRAR_NOMBRE_COLS)
    # Renombra valores de algunas columnas
    for nombre_col, mapa_valores in MAPA_RENOMBRAR_VALORES.items():
        df[nombre_col] = df[nombre_col].map(
            mapa_valores,
            na_action='ignore'
        )
    # Agregar una nueva columna de 
    df[COL_CONCATENAR_COMPLETO_GENERO] = df[COL_COMPLETO] + ' - ' + df[COL_GENERO]
    return df

def agrupar_dataframe_por_columnas_rellenar_con_cero(
    df:Any,
    columnas:list[str],
    nombre_col_total:str,
    llenar_con_cero_faltantes:bool
):
    
    if llenar_con_cero_faltantes is False:
        agregado = df.groupby(columnas).size().reset_index(
            name=nombre_col_total
        ).sort_values(
           by=columnas
        )
        return agregado
    
    # agrupar
    agregado = df.groupby(columnas).size().reset_index(name=nombre_col_total)

    # Crear multiindex con todas las posibles combinaciones
    valores_unicos = []
    for col in columnas:
        unicos = df[col].unique()
        valores_unicos.append(unicos)
        
    multi_index = pd.MultiIndex.from_product(
        valores_unicos,
        names=columnas
    )

    # reindexar y llenar valores que no esten en el multiindex  con 0
    agregado = agregado.set_index(columnas).reindex(multi_index, fill_value=0).reset_index()

    # ordenar el resultado
    agregado = agregado.sort_values(by=columnas)
    
    return agregado


def agregar_totales_a_distplot_stack(
    g:Any,
    total_hue:int
):
    # Access the axes from the FacetGrid
    axes = g.axes.flatten()
    total_p = len(axes[0].patches)
    total_barras = int(total_p/total_hue)
    p_indice_ultima_barra = total_p - total_barras
    alturas_barra = [0 for _ in range(0,total_barras)]
    for ax in axes:                   
        for p_indice, p in enumerate(ax.patches):
            barra_indice = p_indice % total_barras
            altura = alturas_barra[barra_indice-1] = alturas_barra[barra_indice-1] + p.get_height()
            # pon el valor total
            if p_indice>=p_indice_ultima_barra:
                x = p.get_x() + p.get_width() / 2
                ax.annotate(f'{int(altura)}',
                            xy=(x, altura),
                            xytext=(0, 3),  # Vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                
def agrega_totales_a_catplot(
    g:Any
):
    total = len(g.axes[0])
    # Coloca el número de personas por cada barra
    for i in range(0,total):
        ax = g.axes[0,i]
        for i in ax.containers:
            ax.bar_label(i,fmt='%d')    
    
    
def grafica_stack(
    df:Any,
    ruta_archivo:str,
    x_col:str,
    hue_col:str,
    x_label:str,
    y_label:str,
    titulo:Optional[str]
):      
    g=sns.displot(df, x=x_col, hue=hue_col, multiple="stack",palette=PALETA_COLORES)
    g.set_axis_labels(x_label, y_label)  
    total_hue = len(df[hue_col].unique())
    agregar_totales_a_distplot_stack(g=g,total_hue=total_hue)       
    # muestra los bordes: arriba, derecha, abajo e izquierda 
    sns.despine(top = False, right=False, bottom=False, left=False)    
    #plt.show()
    # guarda imagen
    if titulo is not None:
        plt.title(titulo)
    plt.savefig(ruta_archivo)
    plt.close()
    
def grafica_barras(
    df:Any,
    ruta_archivo:str,
    x_col:str,
    y_col:str,    
    x_label:str,
    y_label:str,
    titulos:list[str],
    hue_col:Optional[str],
    col_col:Optional[str],
    color:Optional[tuple[float,float,float]]=None    
):    
    # construye gráfica
    sns.set_palette(PALETA_COLORES)
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
    # etiquetas eje X y Y
    g.set_axis_labels(x_label, y_label)     
    #g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    for indice, titulo in enumerate(titulos):
        # Nombre fibura 0 (izq) y 1 (derecha)
        g.axes[0,indice].set_title(titulo)
    
    # Coloca el número de personas por cada barra
    agrega_totales_a_catplot(g) 

    # muestra los bordes: arriba, derecha, abajo e izquierda 
    sns.despine(top = False, right=False, bottom=False, left=False)    
    #plt.show()
    # guarda imagen
    plt.savefig(ruta_archivo)
    plt.close()

def grafica_barras_por_agrupamiento(    
    df:Any,
    ruta_archivo:str,
    columnas:list[str],
    nombre_col_total:str,
    llenar_con_cero_faltantes:bool,
    x_col:str,
    y_col:str,
    hue_col:str,
    col_col:Optional[str],
    x_label:str,
    y_label:str,
    titulos:list[str]
):    
    # construye dataframe con los datos agregados
    df = agrupar_dataframe_por_columnas_rellenar_con_cero(
        df=df,
        columnas=columnas,
        nombre_col_total=nombre_col_total,
        llenar_con_cero_faltantes=llenar_con_cero_faltantes
    )        
    # construye grafica
    grafica_barras(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=x_col,
        y_col=y_col,
        hue_col=hue_col,
        col_col=col_col,
        x_label=x_label,
        y_label=y_label,
        titulos=titulos
    )
    

def grafica_disciplina_genero_barras_comp_inc(
    df:Any,
    ruta_archivo:str
):    
    grafica_barras_por_agrupamiento(
        df = df,
        ruta_archivo = ruta_archivo,
        columnas = [COL_DISCIPLINA,COL_COMPLETO,COL_GENERO],
        nombre_col_total=COL_TOTAL,
        llenar_con_cero_faltantes=True,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL,
        hue_col=COL_GENERO,
        col_col=COL_COMPLETO,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulos=[TITULO_INCOMPLETOS,TITULO_COMPLETOS]
    )
    
def grafica_disciplina_genero_barras_comp(
    df:Any,
    ruta_archivo:str
):    
    df = df[df[COL_COMPLETO]==VALOR_COMPLETO]
    grafica_barras_por_agrupamiento(
        df = df,
        ruta_archivo = ruta_archivo,
        columnas = [COL_DISCIPLINA,COL_GENERO],
        nombre_col_total=COL_TOTAL,
        llenar_con_cero_faltantes=True,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL,
        hue_col=COL_GENERO,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulos=[TITULO_COMPLETOS]
    )
    
    
def grafica_disciplina_genero_barras_inc(
    df:Any,
    ruta_archivo:str
):    
    df = df[df[COL_COMPLETO]==VALOR_INCOMPLETO]
    grafica_barras_por_agrupamiento(
        df = df,
        ruta_archivo = ruta_archivo,
        columnas = [COL_DISCIPLINA,COL_GENERO],
        nombre_col_total=COL_TOTAL,
        llenar_con_cero_faltantes=True,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL,
        hue_col=COL_GENERO,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulos=[TITULO_INCOMPLETOS]
    )
    
def grafica_disciplina_genero_stack_comp_inc(
    df:Any,
    ruta_archivo:str
):
    grafica_stack(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        hue_col=COL_CONCATENAR_COMPLETO_GENERO,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulo=TITULO_COMPLETOS_E_INCOMPLETOS
    )                
    
def grafica_disciplina_genero_stack_comp(
    df:Any,
    ruta_archivo:str
):
    df = df[df[COL_COMPLETO]==VALOR_COMPLETO]
    grafica_stack(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        hue_col=COL_GENERO,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulo=TITULO_COMPLETOS
    )                

def grafica_disciplina_genero_stack_inc(
    df:Any,
    ruta_archivo:str
):
    df = df[df[COL_COMPLETO]==VALOR_INCOMPLETO]
    grafica_stack(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        hue_col=COL_GENERO,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulo=TITULO_INCOMPLETOS
    )                
        
    
def grafica_disciplina_afiliacion_barras_comp(
    df:Any,
    ruta_archivo:str
):  
    df = df[df[COL_COMPLETO]==VALOR_COMPLETO]
    # construye dataframe con los datos agregados
    df = agrupar_dataframe_por_columnas_rellenar_con_cero(
        df=df,
        columnas=[COL_DISCIPLINA,COL_AFILIACION],
        nombre_col_total=COL_TOTAL,
        llenar_con_cero_faltantes=True
    )    
    
    grafica_barras(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL,
        hue_col=COL_AFILIACION,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulos=[TITULO_COMPLETOS]
    )

def grafica_disciplina_afiliacion_stack_comp(
    df:Any,
    ruta_archivo:str
):  
    df = df[df[COL_COMPLETO]==VALOR_COMPLETO]    
    grafica_stack(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        hue_col=COL_AFILIACION,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PERSONAS,
        titulo=TITULO_COMPLETOS
    )                
 

def grafica_disciplina_papers_barras_autores_coautores(
    df:Any,
    ruta_archivo:str
):          
    grafica_barras(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL_PUBLICACIONES,
        hue_col=COL_PUBLICACIONES,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PUBLICACIONES,
        #titulos=["Publicaciones SNII y coautores sin SNII"]
        titulos = ["SNII publications and coauthors without SNII"]
    )
    
def grafica_disciplina_papers_barras_autores(
    df:Any,
    ruta_archivo:str
):          
    df = df[df[COL_PUBLICACIONES] == COL_PUBLICACIONES_VALOR_AUTORES_CON_SNII]
    paleta_cols = PALETA_COLORES#sns.color_palette()

    # Identifica el color para la segunda barra (naranja en el tema por defecto)
    color_naranja = paleta_cols[1]
    grafica_barras(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL_PUBLICACIONES,
        hue_col=None,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PUBLICACIONES,
        #titulos=["Publicaciones SNII"],
        titulos = ["SNII publications"],
        color=color_naranja
    )
    
def grafica_disciplina_papers_barras_coautores(
    df:Any,
    ruta_archivo:str
):          
    df = df[df[COL_PUBLICACIONES] == COL_PUBLICACIONES_VALOR_COAUTORES_SIN_SNII]
    grafica_barras(
        df=df,
        ruta_archivo=ruta_archivo,
        x_col=COL_DISCIPLINA,
        y_col=COL_TOTAL_PUBLICACIONES,
        hue_col=None,
        col_col=None,
        x_label=X_LABEL_DISCIPLINA,
        y_label=Y_LABEL_TOTAL_PUBLICACIONES,
        #titulos=["Publicaciones coautores sin SNII"]
        titulos = ["Co-authors without SNII publications"]
    )
      

    
def construye_imagenes_finales():    
    sns.set_theme(style="whitegrid")
    if not os.path.exists(DIRECTORIO_REPORTES):
        os.makedirs(DIRECTORIO_REPORTES)
    df = construir_dataframe()
    
    ####################################### GENERO POR DISCIPLINA ######################################
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-barras_comp." + FORMATO_FIGURA
    grafica_disciplina_genero_barras_comp(
        df=df,
        ruta_archivo=ruta_archivo
    )
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
    
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-barras_inc." + FORMATO_FIGURA
    grafica_disciplina_genero_barras_inc(
        df=df,
        ruta_archivo=ruta_archivo
    )
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
        
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-barras_inc_comp." + FORMATO_FIGURA
    grafica_disciplina_genero_barras_comp_inc(
        df=df,
        ruta_archivo=ruta_archivo
    )
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
    # ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-stack_comp_inc." + FORMATO_FIGURA
    # grafica_disciplina_genero_stack_comp_inc(
    #     df=df,
    #     ruta_archivo=ruta_archivo
    # )
    # ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-stack_comp." + FORMATO_FIGURA
    # grafica_disciplina_genero_stack_comp(
    #     df=df,
    #     ruta_archivo=ruta_archivo
    # )
    # ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_GENERO_POR_DISCIPLINA + "-stack_inc." + FORMATO_FIGURA
    # grafica_disciplina_genero_stack_inc(
    #     df=df,
    #     ruta_archivo=ruta_archivo
    # )
    
    ####################################### AFILIACION POR DISCIPLINA ######################################
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_AFILIACION_POR_DISCIPLINA + "-barras_comp." + FORMATO_FIGURA
    grafica_disciplina_afiliacion_barras_comp(
        df=df,
        ruta_archivo=ruta_archivo
    )    
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
        
    # ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_AFILIACION_POR_DISCIPLINA + "-stack_comp." + FORMATO_FIGURA
    # grafica_disciplina_afiliacion_stack_comp(
    #     df=df,
    #     ruta_archivo=ruta_archivo
    # )
    # df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
    ####################################### PAPERS POR DISCIPLINA ######################################
    # Crear DataFrame
    df2 = pd.DataFrame(TOTAL_PAPERS_POR_AUTOR_Y_COAUTOR)    
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_PAPERS_POR_DISCIPLINA + "-barras_autores_coautores." + FORMATO_FIGURA        
    grafica_disciplina_papers_barras_autores_coautores(
        df=df2,
        ruta_archivo=ruta_archivo
    )    
    df2.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
    ####################################### PAPERS POR DISCIPLINA ######################################
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_PAPERS_POR_DISCIPLINA + "-barras_autores." + FORMATO_FIGURA        
    grafica_disciplina_papers_barras_autores(
        df=df2,
        ruta_archivo=ruta_archivo
    )    
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))
    ####################################### PAPERS POR DISCIPLINA ######################################
    ruta_archivo = PREFIJO_ARCHIVO_IMAGEN_PAPERS_POR_DISCIPLINA + "-barras_coautores." + FORMATO_FIGURA        
    grafica_disciplina_papers_barras_coautores(
        df=df2,
        ruta_archivo=ruta_archivo,
    )    
    df.to_csv(ruta_archivo.replace(FORMATO_FIGURA,"csv"))

 

#############################################################################################################
#  EN SUS MARCAS, LISTXS, FUERA!!
#############################################################################################################
construye_imagenes_finales()
