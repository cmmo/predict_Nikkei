FROM python:3.12-slim

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1
# Install uv
RUN pip install --no-cache-dir uv

COPY requirements.txt .

RUN uv pip install --no-cache-dir -r requirements.txt
    
COPY . .

CMD ["uv", "run", "Predict_Nikkei.py"] 
