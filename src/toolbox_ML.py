#Funcion Describe

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def describe_df(df):
    """
    Genera un resumen estructurado de las características de cada columna de un DataFrame.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada sobre el cual se desea realizar el análisis de variables.

    Retorna:
    pd.DataFrame: DataFrame resumen con las siguientes métricas por columna:
        - DATA_TYPE: tipo de dato de la variable.
        - MISSINGS (%): porcentaje de valores nulos.
        - UNIQUE_VALUES: número de valores únicos.
        - CARDIN (%): porcentaje de cardinalidad (valores únicos respecto al total).
    """

    # Tipo de dato de cada columna
    data_type = df.dtypes

    # Número de valores nulos por columna
    missings = df.isna().sum()

    # Porcentaje de valores nulos
    missings_pct = (missings / len(df) * 100).round(2)

    # Número de valores únicos por columna
    unique_vals = df.nunique()

    # Porcentaje de cardinalidad
    cardin_pct = (unique_vals / len(df) * 100).round(2)
    
    # Construcción del DataFrame resumen
    resumen = pd.DataFrame({
        "DATA_TYPE": data_type,
        "MISSINGS (%)": missings_pct,
        "UNIQUE_VALUES": unique_vals,
        "CARDIN (%)": cardin_pct
    })

    # Reordenar columnas para facilitar lectura (opcional)
    resumen = resumen[["DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"]]

    # Ordenar por cardinalidad descendente
    resumen = resumen.sort_values("CARDIN (%)", ascending=False)

    return resumen

# Función tipifica_variables


def tipifica_variables(df):
    """
    Clasifica automáticamente las variables de un DataFrame según su cardinalidad,
    tipo de dato y porcentaje de cardinalidad.
    """

    resultado = []
    n_filas = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_card = (cardinalidad / n_filas) * 100
        es_numerica = pd.api.types.is_numeric_dtype(df[col])

        # Binaria
        if cardinalidad == 2:
            tipo = "Binaria"

        # Categórica (no numérica o cardinalidad baja)
        elif not es_numerica:
            tipo = "Categórica"

        # Numérica continua (criterios automáticos)
        elif cardinalidad > 30 or porcentaje_card > 5:
            tipo = "Numerica Continua"

        # Numérica discreta
        else:
            tipo = "Numerica Discreta"

        resultado.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultado)

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de variables numéricas del dataframe cuya correlación
    (en valor absoluto) con la variable objetivo es superior a un umbral dado.
    Opcionalmente, filtra además por significación estadística.
    Argumentos:
    df (pandas.DataFrame): DataFrame que contiene las variables predictoras y el target.
    target_col (str): Nombre de la columna objetivo del modelo de regresión.
                      Debe ser una variable numérica continua o de alta cardinalidad.
    umbral_corr (float): Umbral mínimo de correlación absoluta. Debe estar entre 0 y 1.
    pvalue (float, opcional): Nivel de significación del test de hipótesis.
                              Si no es None, sólo se seleccionan las variables
                              cuya correlación sea estadísticamente significativa
                              con significación mayor o igual a 1 - pvalue.
    Retorna:
    list: Lista con los nombres de las columnas numéricas que cumplen los criterios
          de correlación y, si aplica, significación estadística. Devuelve None si
          los argumentos de entrada no son válidos.
    """
    
    # Checks iniciales
    
    # Check dataframe
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas DataFrame.")
        return None
    # Check target_col
    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el dataframe.")
        return None
    # Check que target_col sea numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una variable numérica.")
        return None
    # Check cardinalidad del target (evitar variables categóricas numéricas)
    if df[target_col].nunique() < 10:
        print("Error: target_col no parece una variable numérica continua o de alta cardinalidad.")
        return None
    # Check umbral_corr
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe ser un float entre 0 y 1.")
        return None
    # Check pvalue
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
            print("Error: pvalue debe ser None o un float entre 0 y 1.")
            return None
    
    # Selección de columnas numéricas
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Eliminamos el target de la lista
    num_cols.remove(target_col)
    features_num = []
   
    # Cálculo de correlaciones
   
    for col in num_cols:
        corr, p_val = pearsonr(df[target_col], df[col])
        # Check del umbral de correlación
        if abs(corr) > umbral_corr:
            if pvalue is None:
                features_num.append(col)
            else:
                # Test de hipótesis
                if p_val <= pvalue:
                    features_num.append(col)
    return features_num


import seaborn as sns
import matplotlib.pyplot as plt


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera pairplots entre la variable objetivo y un conjunto de variables numéricas
    que cumplen un umbral mínimo de correlación y, opcionalmente, significación estadística.

    Argumentos:
    df (pd.DataFrame): DataFrame con las variables predictoras y el target.
    target_col (str): Nombre de la columna objetivo del modelo de regresión.
    columns (list): Lista de nombres de columnas a evaluar.
    umbral_corr (float): Umbral mínimo de correlación absoluta (entre 0 y 1).
    pvalue (float, opcional): Nivel de significación estadística.

    Retorna:
    list: Lista de columnas que cumplen las condiciones.
    """


# Checks iniciales


 # Comprobamos que df sea un DataFrame

    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas DataFrame.")
        return None

 # Comprobamos que target_col exista en el DataFrame

    if target_col == "" or target_col not in df.columns:
        print("Error: target_col debe ser una columna válida del dataframe.")
        return None
    
 # Comprobamos que target_col sea numérica

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una variable numérica.")
        return None
    
 # Comprobamos que tenga suficiente cardinalidad

    if df[target_col].nunique() < 10:
        print("Error: target_col no parece una variable numérica continua.")
        return None
    
 # Comprobamos el umbral de correlación

    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe estar entre 0 y 1.")
        return None
 
 # Comprobamos el pvalue

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
            print("Error: pvalue debe ser None o un float entre 0 y 1.")
            return None


# Selección de columnas numéricas


 # Obtenemos columnas numéricas

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

 # Eliminamos el target de la lista

    num_cols.remove(target_col)

 # Si no se pasan columnas, usamos todas las numéricas

    if columns:
        cols_to_eval = columns
    else:
        cols_to_eval = num_cols

    selected_features = []


# Cálculo de correlaciones


 # Saltamos columnas que no sean numéricas

    for col in cols_to_eval:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

 # Eliminamos valores nulos

        df_aux = df[[target_col, col]].dropna()

        if len(df_aux) < 2:
            continue

 # Calculamos correlación de Pearson

        corr, p_val = pearsonr(df_aux[target_col], df_aux[col])

 # Comprobamos el umbral de correlación

        if abs(corr) >= umbral_corr:
            if pvalue is None:
                selected_features.append(col)
            else:
                if p_val <= pvalue:
                    selected_features.append(col)

    if not selected_features:
        print("No hay variables que cumplan los criterios.")
        return []


# Generación de pairplots (máx 5 variables)


    all_columns = [target_col] + selected_features
    max_vars = 5

 # Dividimos en bloques de máximo 5 variables

    blocks = [all_columns[i:i + max_vars] for i in range(0, len(all_columns), max_vars - 1)]

    for block in blocks:
        if target_col not in block:
            block = [target_col] + block

 # Pintamos el pairplot
 
        sns.pairplot(df[block].dropna())
        plt.show()

    return selected_features



# IMPORTACIÓN DE LIBRERÍAS NECESARIAS

import pandas as pd                     # Manipulación de datos
import numpy as np                      # Operaciones numéricas
from scipy.stats import f_oneway, ttest_ind   # ANOVA y t-test
import matplotlib.pyplot as plt         # Gráficos opcionales


# FUNCIÓN: get_features_cat_regression

def get_features_cat_regression(df, target_col, pvalue=0.05, columns=None, with_individual_plot=False):
    """
    Evalúa la relación estadística entre variables categóricas y una variable objetivo numérica
    dentro de un DataFrame. Selecciona automáticamente el test adecuado (t-test o ANOVA)
    según el número de categorías de cada variable. Devuelve una lista con las columnas
    categóricas que presentan una relación estadísticamente significativa con el target.

    Esta función es útil para procesos de selección de características (feature selection)
    en problemas de regresión, permitiendo identificar qué variables categóricas tienen
    un impacto significativo sobre la variable objetivo.

    Argumentos:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene la variable objetivo y las variables categóricas a evaluar.

    target_col : str
        Nombre de la columna objetivo. Debe ser numérica continua o discreta con alta cardinalidad.

    pvalue : float, opcional (default = 0.05)
        Nivel de significación estadística. Una variable se considera significativa si su
        p-value es menor o igual a este valor.

    columns : list, opcional (default = None)
        Lista de columnas categóricas a evaluar. Si es None, se seleccionan automáticamente
        todas las columnas categóricas del DataFrame.

    with_individual_plot : bool, opcional (default = False)
        Si es True, se generan histogramas del target agrupados por categoría para cada
        variable significativa.

    Retorna:
    --------
    list
        Lista con los nombres de las columnas categóricas que muestran relación significativa
        con el target. Si ocurre un error en los parámetros de entrada, retorna None.
    """

    # VALIDACIONES DE ENTRADA

    # Comprobar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None

    # Comprobar que target_col existe
    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobar que target_col es numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérico.")
        return None

    # Comprobar cardinalidad mínima del target
    if df[target_col].nunique() < 10:
        print("Error: target_col no tiene suficiente cardinalidad para ser un target de regresión.")
        return None

    # Comprobar pvalue
    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None

    # Comprobar columns
    if columns is not None and not isinstance(columns, list):
        print("Error: columns debe ser una lista o None.")
        return None

    # SELECCIÓN AUTOMÁTICA DE VARIABLES CATEGÓRICAS

    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    selected_features = []  # Lista donde guardaremos las columnas significativas

    # EVALUACIÓN ESTADÍSTICA DE CADA VARIABLE CATEGÓRICA

    for col in columns:

        # Saltar columnas con más del 50% de nulos
        if df[col].isna().mean() > 0.5:
            continue

        # Categorías únicas sin nulos
        categories = df[col].dropna().unique()

        # Crear grupos del target por categoría
        groups = [df[df[col] == cat][target_col].dropna() for cat in categories]

        # Necesitamos al menos dos grupos
        if len(groups) < 2:
            continue

        # Saltar si algún grupo está vacío
        if any(len(g) == 0 for g in groups):
            continue

        # SELECCIÓN AUTOMÁTICA DEL TEST ESTADÍSTICO

        if len(groups) == 2:
            # t-test para dos categorías
            stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
        else:
            # ANOVA para más de dos categorías
            stat, p_val = f_oneway(*groups)

        # COMPROBACIÓN DE SIGNIFICACIÓN

        if p_val <= pvalue:
            selected_features.append(col)

            # Gráficos opcionales
            if with_individual_plot:
                plt.figure(figsize=(8, 5))
                for cat in categories:
                    plt.hist(df[df[col] == cat][target_col],
                             bins=20, alpha=0.5, label=str(cat))
                plt.title(f"Distribución de {target_col} por categoría de {col}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.legend()
                plt.show()

    # MOSTRAR RESULTADO DIRECTAMENTE

    print("Columnas significativas:", selected_features)

    return selected_features


# IMPORTACIÓN DE LIBRERÍAS NECESARIAS


import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt


# FUNCIÓN: plot_features_cat_regression


def plot_features_cat_regression(df, target_col="", columns=None, pvalue=0.05, with_individual_plot=False):
    """
    Evalúa la relación estadística entre un conjunto de variables (categóricas o numéricas)
    y una variable objetivo numérica continua. La función genera histogramas agrupados
    únicamente para aquellas variables cuyo test estadístico (t-test o ANOVA) resulte
    significativo según el nivel de significación indicado.

    Esta función está diseñada para análisis exploratorio y selección de características
    en problemas de regresión, permitiendo visualizar cómo varía el target según los
    valores de cada variable categórica o numérica discretizada.

    Argumentos:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene la variable objetivo y las variables a evaluar.

    target_col : str
        Nombre de la columna objetivo. Debe ser numérica continua o discreta con alta cardinalidad.

    columns : list, opcional (default = None)
        Lista de columnas a evaluar. Si es None o lista vacía, se seleccionarán automáticamente
        las columnas NUMÉRICAS del DataFrame. Si contiene valores, se evaluarán únicamente
        esas columnas (tratadas como categóricas).

    pvalue : float, opcional (default = 0.05)
        Nivel de significación estadística. Una variable se considera significativa si su
        p-value es menor o igual a este valor.

    with_individual_plot : bool, opcional (default = False)
        Si es True, se generan histogramas del target agrupados por categoría para cada
        variable significativa.

    Retorna:
    --------
    list
        Lista con los nombres de las columnas que presentan relación significativa con el target.
        La función imprime directamente la lista final.
    """

    # VALIDACIONES DE ENTRADA

    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérico.")
        return None

    if df[target_col].nunique() < 10:
        print("Error: target_col no tiene suficiente cardinalidad para regresión.")
        return None

    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None

    if columns is not None and not isinstance(columns, list):
        print("Error: columns debe ser una lista o None.")
        return None

    # SELECCIÓN AUTOMÁTICA DE COLUMNAS

    # Si columns está vacío o None → usar NUMÉRICAS
    if not columns:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
        columns.remove(target_col)  # Evitar usar el target como predictor

    selected_features = []

    # EVALUACIÓN ESTADÍSTICA
    
    for col in columns:

        # Saltar columnas con demasiados nulos
        if df[col].isna().mean() > 0.5:
            continue

        # Convertir numéricas a "pseudo-categóricas" si tienen demasiados valores únicos
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
            continue  # No tiene sentido hacer histogramas por categoría con 200 valores

        # Obtener categorías únicas
        categories = df[col].dropna().unique()

        # Crear grupos del target
        groups = [df[df[col] == cat][target_col].dropna() for cat in categories]

        if len(groups) < 2:
            continue

        if any(len(g) == 0 for g in groups):
            continue

        # Selección automática del test
        if len(groups) == 2:
            stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
        else:
            stat, p_val = f_oneway(*groups)

        # Si es significativa → guardar y graficar
        if p_val <= pvalue:
            selected_features.append(col)

            if with_individual_plot:
                plt.figure(figsize=(8, 5))
                for cat in categories:
                    plt.hist(df[df[col] == cat][target_col],
                             bins=20, alpha=0.5, label=str(cat))
                plt.title(f"Distribución de {target_col} por categoría de {col}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.legend()
                plt.show()
    
    # MOSTRAR RESULTADO DIRECTAMENTE

    print("Columnas significativas:", selected_features)

    return selected_features
