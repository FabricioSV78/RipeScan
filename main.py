from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from catboost import CatBoostClassifier
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import uuid
import time
from datetime import datetime
from scipy.stats import skew, kurtosis
import psycopg2
from psycopg2 import sql
import threading
import os

app = FastAPI()

# Cargar modelos
modelo = CatBoostClassifier()
modelo.load_model("modelo_catboost_mejorado.cbm")
detector = YOLO("yolov9c.pt")


# Configuración de PostgreSQL
def guardar_metricas_postgres(timestamp, tiempo, precision, costo, total):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()

    insert_query = sql.SQL("""
        INSERT INTO metricas_clasificacion (timestamp, tiempo_segundos, precision_promedio, costo_soles, total_platanos)
        VALUES (%s, %s, %s, %s, %s)
    """)
    cursor.execute(insert_query, (timestamp, tiempo, precision, costo, total))

    conn.commit()
    cursor.close()
    conn.close()

# Costo de clasificación
COSTO_POR_SEGUNDO = 0.05

# Extracción de características

def extraer_caracteristicas_mejoradas(arr):
    arr = arr / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    features = []
    for canal in [r, g, b, s, v]:
        flat = canal.flatten()
        features.extend([
            np.mean(flat), np.std(flat), np.min(flat), np.max(flat),
            np.median(flat), skew(flat), kurtosis(flat)
        ])
        hist, _ = np.histogram(flat, bins=8, range=(0, 1))
        features.extend(hist / np.sum(hist))

    return features

# Endpoint de predicción
@app.post("/predict/")
async def predecir_madurez(file: UploadFile = File(...)):
    start_time = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    temp_path = "temp.jpg"
    image.save(temp_path)

    results = detector(temp_path, imgsz=960, verbose=False)
    draw = ImageDraw.Draw(image)

    resultados = []
    total_platanos = 0

    for result in results:
        for box, cls_id, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            clase = detector.model.names[int(cls_id)].lower()
            if clase == "banana" and score >= 0.5:
                total_platanos += 1
                x1, y1, x2, y2 = map(int, box)
                crop = image_np[y1:y2, x1:x2]
                crop = cv2.resize(crop, (100, 100))
                features = extraer_caracteristicas_mejoradas(crop)
                pred = str(modelo.predict([features])[0])

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1 + 5, y1 - 15), pred, fill="red")

                resultados.append({
                    "posicion": [x1, y1, x2, y2],
                    "madurez": pred
                })

    if not resultados:
        return JSONResponse(content={"mensaje": "No se detectó plátano en la imagen."})

    # Guardar imagen anotada
    nombre_imagen = f"resultado_{uuid.uuid4().hex[:6]}.jpg"
    image.save(nombre_imagen)

    # Calcular métricas
    tiempo_total = round(time.time() - start_time, 3)
    precision_aproximada = 1.0  
    costo = round(tiempo_total * COSTO_POR_SEGUNDO, 4)

    # Guardar en PostgreSQL
    guardar_metricas_postgres(
        timestamp=datetime.now(),
        tiempo=tiempo_total,
        precision=precision_aproximada,
        costo=costo,
        total=total_platanos
    )

    return {
        "resultados": resultados,
        "imagen_url": f"/imagen/{nombre_imagen}"
    }

@app.get("/imagen/{nombre}")
def obtener_imagen(nombre: str):
    return FileResponse(nombre, media_type="image/jpeg")



def limpiar_imagenes_antiguas(directorio=".", extensiones=(".jpg", ".png"), max_edad_min=3):
    ahora = time.time()
    for archivo in os.listdir(directorio):
        if archivo.endswith(extensiones):
            ruta = os.path.join(directorio, archivo)
            edad_min = (ahora - os.path.getmtime(ruta)) / 60
            if edad_min > max_edad_min:
                try:
                    os.remove(ruta)
                except Exception:
                    pass

def lanzar_limpieza_periodica(intervalo_min=3):
    def bucle():
        while True:
            limpiar_imagenes_antiguas()
            time.sleep(intervalo_min * 60)
    thread = threading.Thread(target=bucle, daemon=True)
    thread.start()


lanzar_limpieza_periodica()
