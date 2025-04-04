FROM python:3.12-slim

WORKDIR /app

# Use system python
ENV UV_SYSTEM_PYTHON=1

# Install uv
RUN pip install --no-cache-dir uv

COPY requirements.txt .

# Install package by uv
RUN uv pip install --no-cache-dir -r requirements.txt
    
COPY . .

# Run by uv
CMD ["uv", "run", "Predict_Nikkei.py"] 
