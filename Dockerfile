FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_NAME=microsoft/DialoGPT-large
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='$MODEL_NAME')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
