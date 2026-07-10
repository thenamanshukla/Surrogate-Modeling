FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json /app/frontend/package.json
RUN npm install
COPY frontend /app/frontend
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

COPY src /app/src
COPY backend /app/backend
COPY data /app/data
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend:/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
