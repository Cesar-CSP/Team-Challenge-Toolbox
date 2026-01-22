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

'''def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    Argumentos:
    df(pd.DataFrame): DataFrame de entrada con las variables a analizar.
    umbral_categoria (int): Umbral máximo de cardinalidad para considerar una variable como categórica.
    umbral_continua (float): Umbral mínimo de porcentaje de cardinalidad para considerar una variable como continua.

    Retorna:
    pd.DataFrame: DataFrame con dos columnas: 'nombre_variable' y 'tipo_sugerido', indican el tipo asignado a cada variable.
    """
    resultado = []

    n_filas = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_card = (cardinalidad / n_filas) * 100

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_card >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"

        resultado.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultado)'''
    

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
    # --------------------
    # Checks iniciales
    # --------------------
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
    # --------------------
    # Selección de columnas numéricas
    # --------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Eliminamos el target de la lista
    num_cols.remove(target_col)
    features_num = []
    # --------------------
    # Cálculo de correlaciones
    # --------------------
    for col in num_cols:
        corr, p_val = pearsonr(data[target_col], data[col])
        # Check del umbral de correlación
        if abs(corr) > umbral_corr:
            if pvalue is None:
                features_num.append(col)
            else:
                # Test de hipótesis
                if p_val <= pvalue:
                    features_num.append(col)
    return features_num