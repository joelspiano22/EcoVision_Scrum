from __future__ import annotations
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

CATEGORIES = ["Papel", "Cartón", "Plástico", "Vidrio", "Metal"]
SLUGS = ["papel", "carton", "plastico", "vidrio", "metal"]
COLORS = {"Papel":"#77AAFF", "Cartón":"#E7AC69", "Plástico":"#57E6C0", "Vidrio":"#B99AFF", "Metal":"#F6CD69"}
GUIDANCE = {
 "Papel": "Separa el papel limpio y seco. Si está mojado o tiene grasa, consulta las indicaciones del punto de recolección.",
 "Cartón": "Vacía y aplana el cartón. Mantén el material seco y revisa las reglas del punto de recolección.",
 "Plástico": "Vacía el envase y consulta si el punto de recolección acepta ese tipo de plástico. El material por sí solo no garantiza su reciclabilidad.",
 "Vidrio": "Vacía el recipiente. Evita manipular fragmentos con las manos y consulta qué tipos de vidrio acepta el punto de recolección.",
 "Metal": "Vacía las latas o envases. Evita bordes cortantes y consulta las condiciones del punto de recolección.",
}

def now_lima():
    return datetime.now(ZoneInfo("America/Lima")).replace(tzinfo=None)

def open_image(raw: bytes) -> Image.Image:
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Cada fotografía puede ocupar hasta 15 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Utiliza una fotografía JPG o PNG.")
            if source.width * source.height > 20_000_000:
                raise ValueError("La imagen supera 20 megapíxeles. Reduce su tamaño.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            if min(image.size) < 16:
                raise ValueError("La imagen debe medir al menos 16 × 16 píxeles.")
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("No se pudo leer la imagen. Verifica que sea un JPG o PNG válido.") from exc

def image_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

def demo_prediction(image):
    # Resultado reproducible, deliberadamente NO inferencia de material.
    digest = image_hash(image)
    rng = np.random.default_rng(int(digest[:12], 16))
    probabilities = rng.dirichlet(np.ones(5) * .8)
    idx = int(np.argmax(probabilities))
    return {"category": CATEGORIES[idx], "confidence": float(probabilities[idx] * 100),
            "probabilities": dict(zip(CATEGORIES, (probabilities * 100).tolist()))}

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, category TEXT NOT NULL,
                confidence REAL NOT NULL, origin TEXT NOT NULL,
                mode TEXT NOT NULL, image_hash TEXT, probabilities TEXT)""")
    def connect(self):
        return sqlite3.connect(self.path, timeout=15)
    def add(self, prediction, origin, digest, timestamp=None, mode="Simulado"):
        with self.connect() as con:
            cur = con.execute("INSERT INTO analyses(timestamp,category,confidence,origin,mode,image_hash,probabilities) VALUES (?,?,?,?,?,?,?)",
                ((timestamp or now_lima()).isoformat(timespec="seconds"), prediction["category"],
                 prediction["confidence"], origin, mode, digest,
                 json.dumps(prediction.get("probabilities", {}))))
            return cur.lastrowid
    def frame(self):
        with self.connect() as con:
            df = pd.read_sql_query("SELECT * FROM analyses ORDER BY timestamp DESC,id DESC", con)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    def seed(self):
        # Lote único, deliberadamente sintético; jamás métricas de acierto.
        with self.connect() as con:
            if con.execute("SELECT 1 FROM analyses WHERE origin='Lote de ejemplo' LIMIT 1").fetchone():
                return False
            base = now_lima().replace(hour=12, minute=0, second=0, microsecond=0)
            rows = []
            for i in range(20):
                category = CATEGORIES[i % 5]
                confidence = 42 + (i * 17) % 56
                rows.append(((base - timedelta(days=19-i)).isoformat(), category,
                             confidence, "Lote de ejemplo", "Simulado", f"demo-{i}", "{}"))
            con.executemany("INSERT INTO analyses(timestamp,category,confidence,origin,mode,image_hash,probabilities) VALUES (?,?,?,?,?,?,?)", rows)
        return True

def filter_frame(df, categories, confidence, dates=None):
    mask = df.category.isin(categories) & df.confidence.between(*confidence)
    if dates and len(dates) == 2:
        mask &= df.timestamp.dt.date.between(dates[0], dates[1])
    return df.loc[mask].copy()

def visible_frame(df):
    return df[["id", "timestamp", "category", "confidence", "origin", "mode"]].rename(columns={
        "id":"ID", "timestamp":"Fecha y hora (Perú)", "category":"Categoría",
        "confidence":"Confianza (%)", "origin":"Origen", "mode":"Tipo de dato"})
