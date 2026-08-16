FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations
COPY src /app/src
COPY artifacts /app/artifacts

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn gbm_ai.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
