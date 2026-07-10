FROM python:3.11-slim

WORKDIR /app

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY src /app/src
COPY data /app/data
COPY webapp /app/webapp

ENV PYTHONPATH=/app
ENV PORT=8000

CMD ["sh", "-c", "uvicorn webapp.backend.main:app --host 0.0.0.0 --port ${PORT}"]
