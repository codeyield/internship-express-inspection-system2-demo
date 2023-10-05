'''
Асинхронный API backend для упаковки в Docker-контейнер
Выполняет предсказание на обученной модели Yolo8
После запуска приложение начинает слушать URL http://localhost:80/predict/
в ожидании POST-запроса с изображением для детекции

Пример отправки POST-запроса из консоли Windows:
curl -X POST -F "file=@.\path\image.tif" http://localhost:80/predict/

Возвращает json-ответ с результами предсказания
'''
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
import numpy as np
import json
import cv2
import uvicorn

app = FastAPI()

# Загружаем обученную модель
model = YOLO('trained_yolo8s.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # Получаем изображение из POST-запроса
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Предсказание
    predict_results = model(image)

    # Формируем словарь с результатами
    result = predict_results[0]
    predict_data = {"xyxy": result.boxes.xyxy.tolist(),       # список списков xyxy (N, 4)
                    "xywh": result.boxes.xywh.tolist(),       # список списков xywh (N, 4)
                    "xyxyn": result.boxes.xyxyn.tolist(),     # список списков xyxy normalized (N, 4)
                    "xywhn": result.boxes.xywhn.tolist(),     # список списков xywh normalized (N, 4)
                    "conf": result.boxes.conf.tolist(),       # confidence score (N, 1)
                    "cls": result.boxes.cls.tolist(),         # список классов (N, 1)
                    "name": result.names,                     # словарь с именами классов
                    "orig_shape": list(result.orig_shape),    # размеры входного изображения
                    }

    # Возвращаем json ответ
    return json.dumps(predict_data)

# Запускаем приложение и слушаем URL/порт в ожидании POST-запроса с изображением
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)