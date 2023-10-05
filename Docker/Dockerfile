FROM python:3.10-slim-buster

# Installing missing libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY requirements.txt .
COPY trained_yolo8s.pt .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
