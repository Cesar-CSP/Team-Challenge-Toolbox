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

# Función tipifica_variables

def tipifica_variables(df, umbral_categoria, umbral_continua):
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

    return pd.DataFrame(resultado)