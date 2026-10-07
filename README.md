# weatherproject

ML-based Django weather app with forecasting logic in `weatherproject/forecast/`.

## Local setup

1. From repository root, create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Django commands from the nested project directory:
   ```bash
   cd weatherproject
   python manage.py migrate
   python manage.py runserver
   ```
4. Open `http://127.0.0.1:8000/`.

Optional local environment variables:
- `SECRET_KEY` (defaults to a development-only value)
- `DEBUG` (`true` by default)
- `ALLOWED_HOSTS` (comma-separated, defaults to localhost values)
- `CSRF_TRUSTED_ORIGINS` (comma-separated origins)
- `DATABASE_URL` (if not set, SQLite is used)
- `OPENWEATHER_API_KEY` (required for weather API calls)

## Render deployment (Blueprint)

This repository includes `render.yaml` and `build.sh` for deployment.

### What the Blueprint config does
- Creates a Python web service.
- Uses `./build.sh` to install requirements, run `collectstatic`, and apply migrations.
- Starts the app with Gunicorn:
  `cd weatherproject && gunicorn weatherproject.wsgi:application`
- Creates a managed PostgreSQL database and injects `DATABASE_URL`.
- Generates `SECRET_KEY` and sets production env vars (`DEBUG=false`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`).

### Deploy steps
1. Push this repository to GitHub.
2. In Render, choose **New +** → **Blueprint**.
3. Select this repository and deploy.
4. In the Render web service, add `OPENWEATHER_API_KEY` under **Environment Variables** with your OpenWeather API key.
5. Trigger a redeploy so the new environment variable is loaded.
6. After deployment completes, Render provides the live service URL in the dashboard.

## Notes

- `myenv/` is a local committed environment and is not used in deployment.
- Static files are served in production with WhiteNoise after `collectstatic`.
