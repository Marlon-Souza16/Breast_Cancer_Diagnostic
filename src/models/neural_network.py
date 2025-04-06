import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class NeuralNetworkModel:
    def __init__(self, scaler, mlp):
        self.scaler = scaler
        self.mlp = mlp

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict_proba(X_scaled)

def train_neural_network(x_train, y_train, x_val=None, y_val=None):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(8, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.01,
        validation_fraction=0.1,
    )

    mlp.fit(x_train_scaled, y_train)

    if x_val is not None and y_val is not None:
        x_val_scaled = scaler.transform(x_val)
        y_pred_val = mlp.predict(x_val_scaled)
        acc = accuracy_score(y_val, y_pred_val)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred_val))

    return NeuralNetworkModel(scaler, mlp)
