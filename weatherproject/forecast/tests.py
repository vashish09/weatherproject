import os
from unittest.mock import Mock, patch

from django.test import TestCase

from . import views


class ForecastViewTests(TestCase):
    def test_dataset_path_resolves_to_repository_csv(self):
        self.assertEqual(views.WEATHER_DATASET_PATH.name, 'weather.csv')
        self.assertTrue(views.WEATHER_DATASET_PATH.exists())

    def test_openweather_url_uses_env_key_without_hardcoded_literal(self):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'name': 'London',
            'main': {
                'temp': 20,
                'feels_like': 19,
                'temp_min': 18,
                'temp_max': 22,
                'humidity': 70,
                'pressure': 1012,
            },
            'weather': [{'description': 'clear sky'}],
            'sys': {'country': 'GB'},
            'wind': {'deg': 100, 'speed': 4},
            'clouds': {'all': 10},
            'visibility': 10000,
        }

        with patch.dict(os.environ, {views.OPENWEATHER_API_KEY_ENV: 'test-key'}, clear=False):
            with patch('forecast.views.requests.get', return_value=mock_response) as mock_get:
                views.get_current_weather('London')

        called_url = mock_get.call_args[0][0]
        self.assertIn('units=metric', called_url)
        self.assertIn('appid=test-key', called_url)
        self.assertNotIn('83332e8b60f969b5d647ace09737b5aa', called_url)

    def test_missing_api_key_fails_gracefully(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(views.OPENWEATHER_API_KEY_ENV, None)
            with patch('forecast.views.requests.get') as mock_get:
                weather_data = views.get_current_weather('London')

        self.assertIsNone(weather_data)
        mock_get.assert_not_called()
