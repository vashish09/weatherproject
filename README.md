# weatherproject

ML-based Django app predicting weather with historical data and interactive visualizations.

## Render deployment (Python Web Service)

1. Create a new Render **Web Service** from this GitHub repository and use **Python 3**.
2. Set **Build Command** to `./build.sh`.
3. Set **Start Command** to `cd weatherproject && gunicorn weatherproject.wsgi:application`.
4. Add environment variables:
   - `OPENWEATHER_API_KEY` = your API key
   - `ALLOWED_HOSTS` = `.onrender.com,localhost,127.0.0.1` (or your custom host list)
   - `DEBUG` = `False`
   - `SECRET_KEY` = generated secret value (or let Render generate via `render.yaml`)
5. Deploy the service and open the generated `.onrender.com` URL.

## Database note

The project currently uses SQLite fallback by default. This works for basic deployment, but PostgreSQL is recommended for persistent production data (set `DATABASE_URL` in Render when ready).
