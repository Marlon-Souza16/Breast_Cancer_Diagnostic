from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_svm_classifier(X_train, y_train, X_val=None, y_val=None, C=1.0):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True, C=C, max_iter=10000, random_state=42)
    )
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))

    return model
