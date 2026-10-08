import os
from datetime import datetime

import pandas as pd
import pytz
import requests

from django.conf import settings
from django.shortcuts import render

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/"


# =========================================================
# CURRENT WEATHER
# =========================================================

def get_current_weather(city):
    url = f"{BASE_URL}weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        print(
            f"Current weather error: "
            f"{response.status_code} - {response.text}"
        )
        return None

    data = response.json()

    return {
        "city": data["name"],
        "country": data["sys"]["country"],

        # IMPORTANT:
        # units=metric already gives Celsius.
        "current_temperature": round(data["main"]["temp"], 1),
        "feels_like": round(data["main"]["feels_like"], 1),
        "temp_min": round(data["main"]["temp_min"], 1),
        "temp_max": round(data["main"]["temp_max"], 1),

        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],

        "wind_gust_dir": data["wind"].get("deg", 0),
        "wind_gust_speed": data["wind"].get("speed", 0),

        "pressure": data["main"]["pressure"],
        "clouds": data["clouds"]["all"],
        "visibility": data.get("visibility", 0),

        # OpenWeather city timezone offset in seconds
        "timezone_offset": data.get("timezone", 0),
    }


# =========================================================
# REAL OPENWEATHER FORECAST
# =========================================================

def get_forecast(city):
    url = f"{BASE_URL}forecast"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        print(
            f"Forecast error: "
            f"{response.status_code} - {response.text}"
        )
        return []

    data = response.json()

    city_timezone_offset = data.get("city", {}).get("timezone", 0)

    forecast_data = []

    # OpenWeather free forecast gives approximately
    # 3-hour forecast intervals.
    for item in data.get("list", [])[:5]:

        # Convert UTC timestamp to the city's local time
        local_timestamp = item["dt"] + city_timezone_offset
        forecast_time = datetime.utcfromtimestamp(local_timestamp)

        forecast_data.append({
            "time": forecast_time.strftime("%H:%M"),
            "temperature": round(item["main"]["temp"], 1),
            "humidity": round(item["main"]["humidity"], 1),
            "description": item["weather"][0]["description"],
        })

    return forecast_data


# =========================================================
# HISTORICAL DATA
# =========================================================

def read_historic_data(file_name):
    df = pd.read_csv(file_name)

    df = df.dropna()
    df = df.drop_duplicates()

    return df


# =========================================================
# PREPARE RAIN MODEL
# =========================================================

def prepare_data(data):
    le = LabelEncoder()

    data = data.copy()

    data["WindGustDir"] = le.fit_transform(
        data["WindGustDir"]
    )

    data["RainTomorrow"] = le.fit_transform(
        data["RainTomorrow"]
    )

    X = data[
        [
            "MinTemp",
            "MaxTemp",
            "WindGustDir",
            "WindGustSpeed",
            "Humidity",
            "Pressure",
            "Temp",
        ]
    ]

    Y = data["RainTomorrow"]

    return X, Y, le


# =========================================================
# TRAIN RAIN MODEL
# =========================================================

def train_rain_model(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print(
        f"Rain Prediction Model Accuracy: {accuracy:.2f}"
    )

    print(
        classification_report(
            Y_test,
            Y_pred,
        )
    )

    return model


# =========================================================
# MAIN WEATHER VIEW
# =========================================================

def weather_view(request):

    if request.method == "POST":

        city = request.POST.get("city", "").strip()

        if not city:
            return render(request, "weather.html")

        # -------------------------------------------------
        # CURRENT WEATHER
        # -------------------------------------------------

        current_weather = get_current_weather(city)

        if current_weather is None:
            return render(request, "weather.html")

        # -------------------------------------------------
        # REAL FORECAST FROM OPENWEATHER
        # -------------------------------------------------

        forecast = get_forecast(city)

        # -------------------------------------------------
        # HISTORICAL DATA + ML RAIN PREDICTION
        # -------------------------------------------------

        csv_path = settings.BASE_DIR.parent / "weather.csv"

        historical_data = read_historic_data(csv_path)

        X, Y, label_encoder = prepare_data(
            historical_data
        )

        rain_model = train_rain_model(X, Y)

        # -------------------------------------------------
        # WIND DIRECTION
        # -------------------------------------------------

        wind_deg = current_weather["wind_gust_dir"] % 360

        compass_points = [
            ("N", 0, 11.25),
            ("NNE", 11.25, 33.75),
            ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75),
            ("E", 78.75, 101.25),
            ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25),
            ("SSE", 146.25, 168.75),
            ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75),
            ("SW", 213.75, 236.25),
            ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25),
            ("WNW", 281.25, 303.75),
            ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75),
            ("N", 348.75, 360),
        ]

        compass_direction = next(
            (
                point
                for point, start, end in compass_points
                if start <= wind_deg < end
            ),
            "N",
        )

        if compass_direction in label_encoder.classes_:
            compass_direction_encoded = (
                label_encoder.transform(
                    [compass_direction]
                )[0]
            )
        else:
            compass_direction_encoded = -1

        # -------------------------------------------------
        # ML INPUT
        # -------------------------------------------------

        current_data = {
            "MinTemp": current_weather["temp_min"],
            "MaxTemp": current_weather["temp_max"],
            "WindGustDir": compass_direction_encoded,
            "WindGustSpeed": current_weather["wind_gust_speed"],
            "Humidity": current_weather["humidity"],
            "Pressure": current_weather["pressure"],
            "Temp": current_weather["current_temperature"],
        }

        current_df = pd.DataFrame([current_data])

        rain_prediction = rain_model.predict(
            current_df
        )[0]

        # -------------------------------------------------
        # FORECAST VALUES
        # -------------------------------------------------

        forecast_values = forecast[:5]

        while len(forecast_values) < 5:
            forecast_values.append({
                "time": "--:--",
                "temperature": "--",
                "humidity": "--",
                "description": "unavailable",
            })

        time1 = forecast_values[0]["time"]
        time2 = forecast_values[1]["time"]
        time3 = forecast_values[2]["time"]
        time4 = forecast_values[3]["time"]
        time5 = forecast_values[4]["time"]

        temp1 = forecast_values[0]["temperature"]
        temp2 = forecast_values[1]["temperature"]
        temp3 = forecast_values[2]["temperature"]
        temp4 = forecast_values[3]["temperature"]
        temp5 = forecast_values[4]["temperature"]

        hum1 = forecast_values[0]["humidity"]
        hum2 = forecast_values[1]["humidity"]
        hum3 = forecast_values[2]["humidity"]
        hum4 = forecast_values[3]["humidity"]
        hum5 = forecast_values[4]["humidity"]

        # -------------------------------------------------
        # INDIA TIME FOR PAGE DATE
        # -------------------------------------------------

        india_timezone = pytz.timezone(
            "Asia/Kolkata"
        )

        now = datetime.now(india_timezone)

        # -------------------------------------------------
        # TEMPLATE CONTEXT
        # -------------------------------------------------

        context = {

            "location": current_weather["city"],
            "city": current_weather["city"],
            "country": current_weather["country"],

            # Current weather
            "current_temp": current_weather[
                "current_temperature"
            ],

            "Mintemp": current_weather["temp_min"],
            "Maxtemp": current_weather["temp_max"],

            "feels_like": current_weather[
                "feels_like"
            ],

            "humidity": current_weather["humidity"],
            "clouds": current_weather["clouds"],

            "description": (
                current_weather["description"]
                .replace("-", " ")
                .lower()
            ),

            # Additional information
            "time": now,
            "date": now.strftime("%B %d, %Y"),

            "wind": current_weather[
                "wind_gust_speed"
            ],

            "pressure": current_weather["pressure"],

            "visibility": current_weather[
                "visibility"
            ],

            # ML prediction
            "rain_prediction": (
                "Yes"
                if rain_prediction == 1
                else "No"
            ),

            # REAL OpenWeather forecast
            "time1": time1,
            "time2": time2,
            "time3": time3,
            "time4": time4,
            "time5": time5,

            "temp1": temp1,
            "temp2": temp2,
            "temp3": temp3,
            "temp4": temp4,
            "temp5": temp5,

            "hum1": hum1,
            "hum2": hum2,
            "hum3": hum3,
            "hum4": hum4,
            "hum5": hum5,
        }

        return render(
            request,
            "weather.html",
            context,
        )

    # Initial page load
    return render(
        request,
        "weather.html",
    )
