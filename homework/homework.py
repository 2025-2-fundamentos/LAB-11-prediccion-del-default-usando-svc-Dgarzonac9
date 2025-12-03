# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def preprocess_data(df):
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], inplace=True)
    df['EDUCATION'].replace(0, np.nan, inplace=True)
    df['MARRIAGE'].replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

def split_features_labels(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def build_model_pipeline(X):
    categorical = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical = [col for col in X.columns if col not in categorical]

    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', StandardScaler(), numerical)
        ]
    )

    pipeline = Pipeline([
        ('transform', transformer),
        ('pca', PCA()),
        ('select', SelectKBest(score_func=f_classif)),
        ('classifier', SVC())
    ])

    return pipeline

def tune_hyperparameters(pipeline, X, y):
    param_grid = {
        'pca__n_components': [21],
        'select__k': [12],
        'classifier__C': [0.8],
        'classifier__kernel': ['rbf'],
        'classifier__gamma': [0.1]
    }
    search = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy', cv=10, n_jobs=-1)
    search.fit(X, y)
    return search

def export_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(model, f)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    metrics_list = []
    matrices_list = []
    for name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        metrics = {
            'type': 'metrics',
            'dataset': name,
            'precision': round(precision_score(y, y_pred), 3),
            'balanced_accuracy': round(balanced_accuracy_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1_score': round(f1_score(y, y_pred), 3)
        }
        cm = confusion_matrix(y, y_pred)
        matrix = {
            'type': 'cm_matrix',
            'dataset': name,
            'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }
        metrics_list.append(metrics)
        matrices_list.append(matrix)
    return metrics_list[:1] + metrics_list[1:] + matrices_list[:1] + matrices_list[1:]

def save_metrics(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_json(path, orient='records', lines=True)

def main():
    train_zip = 'files/input/train_data.csv.zip'
    test_zip = 'files/input/test_data.csv.zip'
    model_path = 'files/models/model.pkl.gz'
    metrics_path = 'files/output/metrics.json'

    train_df = pd.read_csv(train_zip, compression='zip')
    test_df = pd.read_csv(test_zip, compression='zip')

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    X_train, y_train = split_features_labels(train_df, 'default')
    X_test, y_test = split_features_labels(test_df, 'default')

    pipeline = build_model_pipeline(X_train)
    model = tune_hyperparameters(pipeline, X_train, y_train)

    export_model(model, model_path)

    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    save_metrics(results, metrics_path)

if __name__ == '__main__':
    main()