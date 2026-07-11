FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch dataset inside the image
RUN mkdir -p data/raw && cd data/raw && \
    curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && \
    unzip ml-100k.zip && rm ml-100k.zip

# Train and register the model at build time so model.pkl exists in the image
RUN python src/training/register_best_model.py

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]