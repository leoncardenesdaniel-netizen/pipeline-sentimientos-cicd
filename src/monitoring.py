import json


THRESHOLDS = {
    "accuracy": 0.20,
    "precision": 0.20,
    "recall": 0.20,
    "f1_score": 0.20
}


def load_metrics(path="metrics/metrics.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def monitor_model(metrics):
    alerts = []

    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric)

        if value is None:
            alerts.append(f"No se encontró la métrica {metric}.")
        elif value < threshold:
            alerts.append(
                f"Alerta: {metric} ha caído por debajo del umbral. "
                f"Valor actual: {value}, umbral: {threshold}"
            )

    return alerts


if __name__ == "__main__":
    metrics = load_metrics()
    alerts = monitor_model(metrics)

    print("Métricas del modelo:")
    print(metrics)

    if alerts:
        print("\nAlertas detectadas:")
        for alert in alerts:
            print(alert)
    else:
        print("\nNo se detectan caídas de rendimiento.")
