import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay
from pandas.plotting import scatter_matrix

logging.basicConfig(
    filename='../utils/app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def read_data(database_id, task_id):
    
    logging.info("==========Starting process id: %s ================", task_id)
    
    breast_cancer_data = fetch_ucirepo(id=database_id)
    x = breast_cancer_data.data.features
    y = breast_cancer_data.data.targets.squeeze()
    logging.info(f"Data balance: {y.value_counts(normalize=True)}")
    print(x, y)
    
    return  x, y

def gen_kfold_and_save(x, filename, n_splits=3, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds_data = []
    for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
        folds_data.append({
            "fold": fold,
            "train_indices": list(train_index),
            "test_indices": list(test_index)
        })

    df_folds = pd.DataFrame(folds_data)
    df_folds.to_csv(filename, index=False)

    return df_folds

def load_kfold_df(filename):
    df_folds = pd.read_csv(filename)
    return df_folds

def plot_graph(graph_type, data, labels=None, cmap='rainbow', title=None):
    plt.figure(figsize=(10, 8))
    
    if graph_type == 'roc':
        for model_name, (fpr, tpr, roc_auc) in data.items():
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('Taxa de Falso Positivo (FPR)')
        plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
        plt.title(title or 'Curva ROC Comparativa')
        plt.legend(loc='lower right')

    elif graph_type == 'accurary':
        plt.bar(labels, data, color=plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(labels))))
        plt.ylabel('Acurácia Média')
        plt.ylim(0, 1)
        plt.title(title or 'Média de Acurácia por Modelo')

    elif graph_type == 'confusion_matrix':
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(data, display_labels=labels)
        disp.plot(cmap='Blues', ax=ax)

        plt.title(title or 'Matriz de Confusão')
        plt.xlabel('Predições do Modelo')
        plt.ylabel('Valores Verdadeiros')

        ax.text(0, 0.3, 'Verdadeiro Negativo', ha='center', va='top', color='white', fontsize=10, weight='bold')
        ax.text(1, 0.3, 'Falso Positivo', ha='center', va='top', color='black', fontsize=10, weight='bold')
        ax.text(0, 1.3, 'Falso Negativo', ha='center', va='bottom', color='black', fontsize=10, weight='bold')
        ax.text(1, 1.3, 'Verdadeiro Positivo', ha='center', va='bottom', color='white', fontsize=10, weight='bold')

    elif graph_type == 'scatter_matrix':
        scatter_matrix(data, figsize=(12, 12), diagonal='kde', color=plt.cm.get_cmap(cmap)(0.5))
        plt.suptitle(title or 'Scatter Matrix dos Dados')

    plt.grid(True)
    plt.tight_layout()
    plt.show()