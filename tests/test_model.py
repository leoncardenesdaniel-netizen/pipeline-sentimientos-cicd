import os
import json
import joblib


def test_model_file_exists():
    """
    Comprueba que el modelo entrenado se ha guardado correctamente.
    """
    assert os.path.exists("models/modelo_sentimiento.joblib")


def test_metrics_file_exists():
    """
    Comprueba que el archivo de métricas se ha generado correctamente.
    """
    assert os.path.exists("metrics/metrics.json")


def test_model_accuracy_threshold():
    """
    Valida que el modelo supera un umbral mínimo de accuracy.
    Este umbral permite detectar caídas importantes de rendimiento.
    """
    with open("metrics/metrics.json", "r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert metrics["accuracy"] >= 0.50


def test_model_can_predict():
    """
    Comprueba que el modelo puede realizar una predicción sobre un texto nuevo.
    """
    model = joblib.load("models/modelo_sentimiento.joblib")

    text = ["La empresa presenta buenos resultados y mejora sus previsiones"]
    prediction = model.predict(text)

    assert prediction[0] in [0, 1]
