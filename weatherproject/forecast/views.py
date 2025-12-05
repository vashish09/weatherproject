# import necessary libraries
import os
from django.shortcuts import render
import re
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
import pytz

API_KEY = '83332e8b60f969b5d647ace09737b5aa'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# current data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metrics"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return {
            'city': data['name'],
            'current_temperature': round(data['main']['temp'] - 273.15),
            'feels_like': round(data['main']['feels_like'] - 273.15),
            'temp_min': round(data['main']['temp_min'] - 273.15),
            'temp_max': round(data['main']['temp_max'] - 273.15),
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'wind_gust_speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    else:
        return None

# read historical data
def read_historic_data(file_name):
    df = pd.read_csv(file_name)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# data for training
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    Y = data['RainTomorrow']
    return X, Y, le

# train rain prediction model
def train_rain_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Rain Prediction Model Accuracy: {accuracy:.2f}")
    print(classification_report(Y_test, Y_pred))

    return model

# regression data preparation
def prepare_regression_data(data, feature):
    X, Y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        Y.append(data[feature].iloc[i + 1])

    x = np.array(X).reshape(-1, 1)
    y = np.array(Y)
    return x, y

# train regression model
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model

# predict future
def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# weather analytics function
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)

        if current_weather is None:
            print(f"Error: Could not retrieve weather data for city '{city}'. Please check the city name.")
            return render(request, 'weather.html')

        csv_path = os.path.join('C:\\Machine learning\\weather.csv')
        historical_data = read_historic_data(csv_path)
        X, Y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, Y)

        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.25, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), None)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction and compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temperature']
        }
        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(x_hum, y_hum)

        future_temp = predict_future(temp_model, current_data['MinTemp'])
        future_hum = predict_future(hum_model, current_data['Humidity'])

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_time = [(next_hour + timedelta(hours=i)).strftime('%H:%M') for i in range(5)]

        # store each value separately
        time1, time2, time3, time4, time5 = future_time
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_hum

        # pass data to template
        context = {
            'location': city,
            'current_temp': current_weather['current_temperature'],
            'Mintemp': current_weather['temp_min'],
            'Maxtemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'].replace('-', ' ').lower(),
            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': datetime.now(),
            'date': datetime.now().strftime('%B %d,%Y'),

            'wind': current_weather['wind_gust_speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],

            'rain_prediction': "Yes" if rain_prediction == 1 else "No",

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            'temp1': f"{round(temp1,1)}",
            'temp2': f"{round(temp2,1)}",
            'temp3': f"{round(temp3,1)}",
            'temp4': f"{round(temp4,1)}",
            'temp5': f"{round(temp5,1)}",

            'hum1': f"{round(hum1,1)}",
            'hum2': f"{round(hum2,1)}",
            'hum3': f"{round(hum3,1)}",
            'hum4': f"{round(hum4,1)}",
            'hum5': f"{round(hum5,1)}",
        }

        return render(request, 'weather.html', context)
    else:
        return render(request, 'weather.html')
