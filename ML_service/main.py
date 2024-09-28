from model import PredictModel
from flask import Flask, request, jsonify
from dataclasses import asdict

app = Flask(__name__)
p_model = PredictModel()


@app.route("/", methods=["GET"])
def index():
    return jsonify({"text": "Интеллектуальный помощник оператора службы поддержки."})


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    try:
        data = request.json
        print(data)

        feedback = p_model.inference(data.get('question'))
        return jsonify(asdict(feedback))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5221)


