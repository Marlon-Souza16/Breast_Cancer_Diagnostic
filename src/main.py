import os
import numpy as np
import logging
import uuid

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report

# Supondo que você tenha um módulo data_process.py com essas funções
import data_process as data

# Supondo que você tenha um módulo models.py com essas três funções
from models import (
    train_bayesian_network,
    train_neural_network,
    train_svm_classifier
)

def main(x, y):
    # Gera e carrega os folds K-Fold
    filename = 'kfold_splits.csv'
    df_splits = data.gen_kfold_and_save(x, filename=filename, n_splits=10, shuffle=True, random_state=42)
    df_splits = data.load_kfold_df(filename)

    # Definição dos modelos disponíveis
    models = {
        "Bayesian": train_bayesian_network,
        "Neural Network": train_neural_network,
        "SVM": train_svm_classifier
    }

    # Dicionário para armazenar acurácias de cada modelo
    models_acc_list = {model_name: [] for model_name in models}

    # Dicionários para armazenar curvas ROC e matrizes de confusão
    roc_curves = {}
    confusion_matrices = {}

    # Loop nos modelos
    for model_name, train_model in models.items():
        logging.info("=== Training and rating model : %s ===", model_name)
        
        # Listas para guardar todas as saídas verdadeiras e os scores
        y_true_all, y_score_all = [], []

        # Loop nos folds do K-Fold
        for _, row in df_splits.iterrows():
            fold = row["fold"]
            train_indices = eval(row["train_indices"])
            test_indices = eval(row["test_indices"])

            # Treina o modelo usando apenas os índices de treino
            X_train = x.iloc[train_indices]
            y_train = y.iloc[train_indices]
            model = train_model(X_train, y_train)

            # Separa dados de teste
            X_test = x.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # Em vez de usar predict (0/1), vamos pegar as probabilidades para a classe 1
            y_proba = model.predict_proba(X_test)[:, 1]

            # Caso queira calcular acurácia, matriz de confusão, etc., defina um limiar (por ex., 0.5)
            y_pred = (y_proba >= 0.5).astype(int)

            # Calcula métricas básicas
            accuracy = accuracy_score(y_test, y_pred)
            models_acc_list[model_name].append(accuracy)

            # Guarda todos os valores para depois gerar a curva ROC (fpr, tpr)
            y_true_all.extend(y_test)
            y_score_all.extend(y_proba)

            logging.info(f"Fold {fold} - Accuracy: {accuracy:.4f}")
            logging.info(classification_report(y_test, y_pred))

        # Depois de todos os folds, calcula a curva ROC usando os scores (probabilidades) armazenados
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        roc_curves[model_name] = (fpr, tpr, auc(fpr, tpr))

        # Matriz de confusão (usando limiar de 0.5)
        y_pred_final = [1 if score >= 0.5 else 0 for score in y_score_all]
        cm = confusion_matrix(y_true_all, y_pred_final)
        confusion_matrices[model_name] = cm

    # Remove o arquivo CSV de splits (se for necessário)
    os.remove(filename)

    # Calcula a acurácia média de cada modelo
    average_acc = [np.mean(models_acc_list[m]) for m in models]

    # Gera gráficos (você deve ter funções específicas em data_process para isso)
    data.plot_graph('accurary', average_acc, labels=list(models.keys()), cmap='rainbow')
    data.plot_graph('roc', roc_curves, cmap='rainbow')

    for model_name, cm in confusion_matrices.items():
        data.plot_graph('confusion_matrix', cm, labels=np.unique(y), cmap='rainbow',
                        title=f'Matriz de Confusão: {model_name}')

if __name__ == "__main__":
    task_id = uuid.uuid4()
    logging.info("==========Starting process id: %s ================", task_id)

    # Exemplo de leitura de dados. Ajuste para sua realidade
    # Retorna X (features) e y (alvo)
    x, y = data.read_data(17, task_id)
    # Mapeia rótulos B -> 0 e M -> 1
    y = y.map({'B': 0, 'M': 1})

    main(x, y)
