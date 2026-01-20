#Funcion Describe

import pandas as pd

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