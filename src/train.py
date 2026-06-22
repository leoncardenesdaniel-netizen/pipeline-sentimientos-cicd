import os
import json
import joblib
import mlflow
import mlflow.sklearn

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Dataset simulado de análisis de sentimientos financieros
texts = [
    "La acción de Apple sube tras presentar buenos resultados",
    "Los inversores muestran confianza en el crecimiento de la empresa",
    "El mercado reacciona positivamente a las previsiones de beneficios",
    "La compañía mejora sus márgenes y aumenta sus ingresos",
    "Los analistas mantienen una recomendación positiva sobre el valor",
    "La empresa supera las expectativas del mercado",
    "El sentimiento de los inversores es optimista",
    "La demanda de productos tecnológicos sigue aumentando",
    "La acción cae por debajo de las expectativas",
    "Los resultados trimestrales decepcionan al mercado",
    "La empresa reduce sus previsiones de crecimiento",
    "Los inversores muestran preocupación por la caída de ingresos",
    "El aumento de costes presiona los márgenes",
    "El mercado reacciona negativamente a las noticias",
    "Los analistas rebajan la recomendación de la compañía",
    "La incertidumbre económica afecta a la valoración",
]

# 1 = sentimiento positivo, 0 = sentimiento negativo
labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    mlflow.set_experiment("modelo_clasificacion_sentimientos")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modelo": "LogisticRegression + TF-IDF",
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(pipeline, "models/modelo_sentimiento.joblib")

        with open("metrics/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        mlflow.sklearn.log_model(pipeline, "modelo_sentimiento")

        print("Modelo entrenado correctamente")
        print(metrics)


if __name__ == "__main__":
    main()
