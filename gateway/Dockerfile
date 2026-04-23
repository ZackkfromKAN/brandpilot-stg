FROM python:3.12-slim

WORKDIR /app

# Install gateway dependencies only — keeps the image small
COPY gateway/requirements.txt ./gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

# Copy gateway package
COPY gateway/ ./gateway/

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
