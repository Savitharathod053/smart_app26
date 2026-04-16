from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

# Fertilizer logic
def recommend_fertilizer(N, P, K):
    if N < 50:
        return "Urea"
    elif P < 50:
        return "DAP"
    elif K < 50:
        return "MOP"
    else:
        return "NPK Fertilizer"

# Weather API
def get_weather(city):
    API_KEY = "1363cc7c140d05f8f8b6762ec443c244"    # 🔴 replace this

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        res = requests.get(url)
        data = res.json()

        if data.get("main"):
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            rainfall = data.get("rain", {}).get("1h", 0)
            return temp, humidity, rainfall
        else:
            return None
    except:
        return None

# Home page
@app.route('/')
def home():
    return render_template("training.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        city = data.get("city")
        N = int(data.get("N"))
        P = int(data.get("P"))
        K = int(data.get("K"))

        weather = get_weather(city)

        if weather is None:
            return jsonify({"error": "Invalid city"}), 400

        temp, humidity, rainfall = weather

        features = np.array([[N, P, K, temp, humidity, rainfall]])
        crop = model.predict(features)[0]

        fertilizer = recommend_fertilizer(N, P, K)

        return jsonify({
            "crop": crop,
            "fertilizer": fertilizer,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)