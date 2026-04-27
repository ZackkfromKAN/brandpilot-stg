FROM python:3.12-slim

WORKDIR /app

# Install gateway + core dependencies
COPY gateway/requirements.txt ./gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

# Copy packages — core/ is needed by gateway/tools.py
COPY core/ ./core/
COPY gateway/ ./gateway/

EXPOSE 8000

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
