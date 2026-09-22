FROM python:3.12-slim

WORKDIR /app
COPY web/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY sim /app/sim
COPY web /app/web
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
