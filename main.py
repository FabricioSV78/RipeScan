from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from catboost import CatBoostClassifier
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import uuid
import time
from datetime import datetime
from scipy.stats import skew, kurtosis
import psycopg2
import urllib.parse as up
import os


app = FastAPI()
recomendaciones_por_madurez = {
    "unripe": "Estos platanos estan verdes. Ideal para freír o dejar madurar unos días.",
    "ripe": "Estos platanos estan maduros. Perfecto para comer al instante o hacer batidos.",
    "overripe": "Estos platanos estan sobremaduros. Úsalo para hacer pan de plátano, postres o compotas."
}

etiquetas_es = {
    "unripe": "Inmaduro",
    "ripe": "Maduro",
    "overripe": "Sobremaduro"
}


# Cargar modelos
modelo = CatBoostClassifier()
modelo.load_model("modelo_catboost_mejorado.cbm")
detector = YOLO("yolov8m.pt")

# Configuración de PostgreSQL


def guardar_metricas_postgres(timestamp, tiempo, precision, costo, total):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("No se encontró la variable de entorno DATABASE_URL")

    # Decodifica la URL si tiene caracteres especiales
    up.uses_netloc.append("postgres")
    url = up.urlparse(db_url)

    conn = psycopg2.connect(
        dbname=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metricas_clasificacion 
        (timestamp, tiempo_segundos, precision_promedio, costo_soles, total_platanos)
        VALUES (%s, %s, %s, %s, %s)
    """, (timestamp, tiempo, precision, costo, total))

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
    agrupados_por_madurez = {
    "unripe": [],
    "ripe": [],
    "overripe": []
    }

    start_time = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    temp_path = "temp.jpg"
    image.save(temp_path)

    results = detector(temp_path, imgsz=960, verbose=False)
    draw = ImageDraw.Draw(image)
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
                pred_raw = modelo.predict([features])
                pred = str(pred_raw[0][0]) if isinstance(pred_raw[0], (list, np.ndarray)) else str(pred_raw[0])
                fuente = ImageFont.truetype("arial.ttf", size=12)

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1 + 5, y1 - 14), etiquetas_es.get(pred, pred), fill="red", font=fuente)

                agrupados_por_madurez[pred].append({
                    "posicion": [x1, y1, x2, y2],
                    "madurez": pred
                })

    respuesta = []
    for madurez, platanos in agrupados_por_madurez.items():
        if platanos:
            respuesta.append({
                "madurez": etiquetas_es.get(madurez, madurez),
                "cantidad": len(platanos),
                "recomendacion": recomendaciones_por_madurez[madurez],
                "platanos": [
                    {
                        "posicion": platano["posicion"],
                        "madurez": etiquetas_es.get(platano["madurez"], platano["madurez"])
                    }
                    for platano in platanos
                ]
            })

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

    if not any(agrupados_por_madurez.values()):
        return {
            "imagen_url": f"/imagen/{nombre_imagen}",
            "resultados": []  # Vacío
        }

    return {
        "imagen_url": f"/imagen/{nombre_imagen}",
        "resultados": respuesta
    }

@app.get("/imagen/{nombre}")
def obtener_imagen(nombre: str):
    return FileResponse(nombre, media_type="image/jpeg")