
#----------------------------------------------
# Se definen las funciones iniciales.
#----------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


# Función imputación de outlier
# ------

def imputar_valores_extremos(df, variable, metodo='media'):
    """
    Imputa valores extremos en una variable de un DataFrame utilizando la media o la mediana.

    Parámetros:
    df (DataFrame): El DataFrame que contiene la variable a imputar.
    variable (str): El nombre de la variable que deseas imputar.
    metodo (str): La forma de imputación ('media' o 'mediana'). Por defecto es 'media'.

    Retorna:
    DataFrame: El DataFrame con la variable imputada.
    """
    if metodo not in ['media', 'mediana']:
        raise ValueError("El método debe ser 'media' o 'mediana'")

    # Calcular la media o la mediana
    if metodo == 'media':
        valor_imputacion = df[variable].mean()
    else:
        valor_imputacion = df[variable].median()

    # Identificar valores extremos (usando una regla de 3 veces la desviación estándar)
    limite_inferior = df[variable].mean() - 3 * df[variable].std()
    limite_superior = df[variable].mean() + 3 * df[variable].std()

    # Imputar valores extremos
    df[variable] = np.where(
        (df[variable] < limite_inferior) | (df[variable] > limite_superior),
        valor_imputacion,
        df[variable]
    )

    return df


# Función imputación perdidos
# ------

def imputar_valores(df, variable, metodo='media', valor_especifico=None):
    """
    Imputa valores perdidos en una columna de un DataFrame según el método especificado.

    Parámetros:
    df (pd.DataFrame): El DataFrame en el que se imputarán los valores.
    variable (str): El nombre de la columna a imputar.
    metodo (str): El método de imputación ('media', 'mediana', 'moda', 'valor_especifico').
    valor_especifico: El valor específico a usar para la imputación (relevante solo si 'metodo' es 'valor_especifico').

    Retorna:
    pd.DataFrame: El DataFrame con la variable imputada.
    """

    if metodo == 'media':
        imputacion = df[variable].mean()
    elif metodo == 'mediana':
        imputacion = df[variable].median()
    elif metodo == 'moda':
        imputacion = df[variable].mode()[0]
    elif metodo == 'valor_especifico':
        if valor_especifico is None:
            raise ValueError("Debe proporcionar un valor específico para la imputación.")
        imputacion = valor_especifico
    else:
        raise ValueError("Método de imputación no reconocido. Use 'media', 'mediana', 'moda' o 'valor_especifico'.")

    df[variable].fillna(imputacion, inplace=True)
    return df


# Funcion graficadora confusion_marix
# ---
def confusion_matrix_graph(cm):
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=['No', 'Yes'],
              yticklabels=['No', 'Yes'])
  plt.title('Matriz de Confusión')
  plt.xlabel('Predicción')
  plt.ylabel('Realidad')
  plt.show()


# Funcion ROC curve
# ---
def roc_curve_graph(y,prob):
  # Obtener las probabilidades de la clase positiva
  y_prob = prob[:, 1]  # Probabilidades de la clase 1

  # Calcular la curva ROC
  fpr, tpr, thresholds = roc_curve(y,  y_prob)

  # Calcular el área bajo la curva (AUC)
  roc_auc = auc(fpr, tpr)

  # Graficar la curva ROC
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
  plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea de referencia
  plt.title('Curva ROC')
  plt.xlabel('Tasa de Falsos Positivos (FPR)')
  plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
  plt.legend(loc='lower right')
  plt.grid()
  plt.show()


#----------------------------------------------
# Script de Preparación de Datos
#----------------------------------------------

import pandas as pd
import numpy as np
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/', filename), sep=';')
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    
    # 1 Limpieza de datos

    ## Separando las variables segun su tipo para un correcta lectura
    # Lista de variables numéricas
    numeric_vars = df.select_dtypes(include=['number']).columns.tolist()
    # Lista de variables categóricas
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Actualizamos variables numericas
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")
    numeric_vars.remove('SeniorCitizen')
    categorical_vars.append('SeniorCitizen')
    numeric_vars, categorical_vars
    # Analizando variables categóricas
    # Iterar sobre las columnas del DataFrame
    for column in df.columns:
     if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        print(f"Resumen de porcentajes para la variable '{column}':\n")
        print(df[column].value_counts(normalize=True) * 100)
        print("\n" + "-"*50 + "\n")

    # Actualizamos variables categoricas
    df["TotalCharges"] = df["TotalCharges"].replace(" ",np.nan)
    df["TotalCharges"] = df["TotalCharges"].astype("float")
    categorical_vars.remove('TotalCharges')
    numeric_vars.append('TotalCharges')

    # Presencia de valores perdidos
    for column in df.columns:
        missing_percentage = df[column].isnull().mean() * 100
        print(f'{column}: {missing_percentage:.2f}%')

    # Tenemos valores en nuestra tabla que son espacios en blanco no necesariamente son valores nulos (TIP OPCIONAL)
    df = imputar_valores(df,'TotalCharges',metodo='mediana')

    for column in df.columns:
        missing_percentage = df[column].isnull().mean() * 100
        print(f'{column}: {missing_percentage:.2f}%')


    # 2 Preprocesamiento de datos

    # Retirando la variable target de la lista de vaiables categoricas
    categorical_vars.remove('Churn')
    categorical_vars.remove('customerID')
    # Guardar todas las variables categoricas en un solo lugar
    cat_cols = df[categorical_vars]
    num_cols = df[numeric_vars]
    # Generar variables para las dos columnas que omiti de mi mapeo de variables cualitativas y cuantitativas
    id_customer = df["customerID"]
    label = df["Churn"]

    # Transformacion de variables categoricas a numericas
    # Label encoding (Target Encoding) : Cambiar en la misma columna el valor categorico a numerico
    # La variable target categorica solo puede ser transformada con el target encoding
    label = label.apply(lambda x: 1 if x == "Yes" else 0) # Yes - 1, No -0


    # Analizando importancia de variables en un modelo simple

    from sklearn.tree import DecisionTreeClassifier
    # Separar las variables de entrada y la target
    X = df[['tenure','TotalCharges']]
    y = df['Churn']
    # Entrenar un árbol de decisión
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X, y)
    # Obtener la importancia de las variables
    importances = tree.feature_importances_
    print(f'Importancia de tenure: {importances[0]}')
    print(f'Importancia de TotalCharges: {importances[1]}')

    # retirando la variable que menos aporta
    numeric_vars.remove('tenure')
    numeric_vars
    del num_cols['tenure']

    #transformamos las variables categóricas a numéricas
    cat_cols = pd.get_dummies(df = cat_cols)
    df = pd.concat([num_cols, cat_cols, label], axis=1)
    
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('Data_Customer_Churn.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['MonthlyCharges',
'TotalCharges',
'gender_Female',
'gender_Male',
'Partner_No',
'Partner_Yes',
'Dependents_No',
'Dependents_Yes',
'PhoneService_No',
'PhoneService_Yes',
'MultipleLines_No',
'MultipleLines_No phone service',
'MultipleLines_Yes',
'InternetService_DSL',
'InternetService_Fiber optic',
'InternetService_No',
'OnlineSecurity_No',
'OnlineSecurity_No internet service',
'OnlineSecurity_Yes',
'OnlineBackup_No',
'OnlineBackup_No internet service',
'OnlineBackup_Yes',
'DeviceProtection_No',
'DeviceProtection_No internet service',
'DeviceProtection_Yes',
'TechSupport_No',
'TechSupport_No internet service',
'TechSupport_Yes',
'StreamingTV_No',
'StreamingTV_No internet service',
'StreamingTV_Yes',
'StreamingMovies_No',
'StreamingMovies_No internet service',
'StreamingMovies_Yes',
'Contract_Month-to-month',
'Contract_One year',
'Contract_Two year',
'PaperlessBilling_No',
'PaperlessBilling_Yes',
'PaymentMethod_Bank transfer (automatic)',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check',
'PaymentMethod_Mailed check',
'SeniorCitizen_0',
'SeniorCitizen_1',
'Churn'],'churn_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('Data_Customer_Churn_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['MonthlyCharges',
'TotalCharges',
'gender_Female',
'gender_Male',
'Partner_No',
'Partner_Yes',
'Dependents_No',
'Dependents_Yes',
'PhoneService_No',
'PhoneService_Yes',
'MultipleLines_No',
'MultipleLines_No phone service',
'MultipleLines_Yes',
'InternetService_DSL',
'InternetService_Fiber optic',
'InternetService_No',
'OnlineSecurity_No',
'OnlineSecurity_No internet service',
'OnlineSecurity_Yes',
'OnlineBackup_No',
'OnlineBackup_No internet service',
'OnlineBackup_Yes',
'DeviceProtection_No',
'DeviceProtection_No internet service',
'DeviceProtection_Yes',
'TechSupport_No',
'TechSupport_No internet service',
'TechSupport_Yes',
'StreamingTV_No',
'StreamingTV_No internet service',
'StreamingTV_Yes',
'StreamingMovies_No',
'StreamingMovies_No internet service',
'StreamingMovies_Yes',
'Contract_Month-to-month',
'Contract_One year',
'Contract_Two year',
'PaperlessBilling_No',
'PaperlessBilling_Yes',
'PaymentMethod_Bank transfer (automatic)',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check',
'PaymentMethod_Mailed check',
'SeniorCitizen_0',
'SeniorCitizen_1',
'Churn'],'churn_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('Data_Customer_Churn_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['MonthlyCharges',
'TotalCharges',
'gender_Female',
'gender_Male',
'Partner_No',
'Partner_Yes',
'Dependents_No',
'Dependents_Yes',
'PhoneService_No',
'PhoneService_Yes',
'MultipleLines_No',
'MultipleLines_No phone service',
'MultipleLines_Yes',
'InternetService_DSL',
'InternetService_Fiber optic',
'InternetService_No',
'OnlineSecurity_No',
'OnlineSecurity_No internet service',
'OnlineSecurity_Yes',
'OnlineBackup_No',
'OnlineBackup_No internet service',
'OnlineBackup_Yes',
'DeviceProtection_No',
'DeviceProtection_No internet service',
'DeviceProtection_Yes',
'TechSupport_No',
'TechSupport_No internet service',
'TechSupport_Yes',
'StreamingTV_No',
'StreamingTV_No internet service',
'StreamingTV_Yes',
'StreamingMovies_No',
'StreamingMovies_No internet service',
'StreamingMovies_Yes',
'Contract_Month-to-month',
'Contract_One year',
'Contract_Two year',
'PaperlessBilling_No',
'PaperlessBilling_Yes',
'PaymentMethod_Bank transfer (automatic)',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check',
'PaymentMethod_Mailed check',
'SeniorCitizen_0',
'SeniorCitizen_1'],'churn_score.csv')
    
if __name__ == "__main__":
    main()
