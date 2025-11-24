
#----------------------------------------------
# Se definen las funciones iniciales.
#----------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os


# Función imputación de outlier
def imputar_valores_extremos(df, variable, metodo='media'):
    if metodo not in ['media', 'mediana']:
        raise ValueError("El método debe ser 'media' o 'mediana'")

    if metodo == 'media':
        valor_imputacion = df[variable].mean()
    else:
        valor_imputacion = df[variable].median()

    limite_inferior = df[variable].mean() - 3 * df[variable].std()
    limite_superior = df[variable].mean() + 3 * df[variable].std()

    df[variable] = np.where(
        (df[variable] < limite_inferior) | (df[variable] > limite_superior),
        valor_imputacion,
        df[variable]
    )
    return df

# Función imputación perdidos
def imputar_valores(df, variable, metodo='media', valor_especifico=None):
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

# Funcion graficadora confusion_matrix
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
def roc_curve_graph(y, prob):
    y_prob = prob[:, 1]
    fpr, tpr, thresholds = roc_curve(y,  y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

#----------------------------------------------
# Script de Preparación de Datos
#----------------------------------------------

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/', filename), sep=';')
    print(filename, ' cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    
    # 1 Limpieza de datos
    numeric_vars = df.select_dtypes(include=['number']).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Transformación correcta de SeniorCitizen
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Solo mostrar qué variables categóricas hay (sin imprimir value_counts gigantes)
    print("Variables categóricas:", categorical_vars)

    # Arreglo TotalCharges como numérica
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)

    # Imputación
    df = imputar_valores(df, 'TotalCharges', metodo='mediana')

    # Preparación para modelado
    categorical_vars = [c for c in categorical_vars if c not in ['Churn','customerID']]
    num_cols = df[numeric_vars]
    cat_cols = df[categorical_vars]

    label = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # Importancia de variables (no cambia)
    from sklearn.tree import DecisionTreeClassifier
    X = df[['tenure','TotalCharges']]
    y = label
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X, y)
    importances = tree.feature_importances_
    print(f'Importancia de tenure: {importances[0]}')
    print(f'Importancia de TotalCharges: {importances[1]}')

    # Retiro tenure
    if 'tenure' in num_cols.columns:
        num_cols = num_cols.drop(columns=['tenure'])

    # One Hot Encoding
    cat_cols = pd.get_dummies(df=cat_cols)

    df = pd.concat([num_cols, cat_cols, label.rename('Churn')], axis=1)

    print("Transformación de datos completa")
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    os.makedirs(os.path.join('../data/processed/'), exist_ok=True)
    dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
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
