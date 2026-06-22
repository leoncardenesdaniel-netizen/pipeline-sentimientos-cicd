import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "models/modelo_sentimiento.joblib"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "API de clasificación de sentimientos activa",
        "modelo": "modelo_sentimiento.joblib"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Debe enviarse un JSON con el campo 'text'"
        }), 400

    text = data["text"]
    prediction = model.predict([text])[0]

    sentimiento = "positivo" if prediction == 1 else "negativo"

    return jsonify({
        "texto": text,
        "prediccion": int(prediction),
        "sentimiento": sentimiento
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
