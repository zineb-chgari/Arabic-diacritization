from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_diacritics, tashkeel_text, load_tashkeel_model

app = Flask(__name__)
CORS(app, resources={r"/predict*": {"origins": "http://localhost:3000"}})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_text = data.get("text", "").strip()

        if not input_text:
            return jsonify({"error": "Empty input text"}), 400

        result = predict_diacritics(input_text)

        return jsonify({"diacritized_text": result})

    except Exception as e:
        app.logger.error(f"Erreur dans /predict : {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
