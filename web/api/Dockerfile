FROM python:3.12-slim

WORKDIR /app
COPY web/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY sim /app/sim
COPY agent /app/agent
COPY web /app/web
ENV PYTHONPATH=/app
ENV HARMONY_MODEL=/models/ckpt_68120.zip
ENV HARMONY_ALT_MODEL=/models/ckpt_35714.zip

EXPOSE 8000
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
