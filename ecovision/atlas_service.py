"""Acceso unificado a MongoDB Atlas y al clasificador EcoVision V1.

Los binarios de imágenes y modelos residen en GridFS. Nunca se registra la URI.
La inferencia se realiza localmente en la app, no en MongoDB ni en Colab.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


class AtlasNotConfigured(RuntimeError):
    """No hay configuración MongoDB disponible."""


def atlas_config():
    """Lee credenciales de st.secrets o variables del entorno, sin divulgarlas."""
    uri = os.environ.get('MONGO_URI') or os.environ.get('MONGODB_URI')
    database = os.environ.get('MONGO_DB') or os.environ.get('MONGODB_DATABASE') or 'ecovision'
    try:
        cfg = st.secrets.get('mongodb', {})
        if not uri:
            uri = cfg.get('uri')
            database = cfg.get('database', database)
    except (FileNotFoundError, KeyError, AttributeError, RuntimeError):
        pass
    if not uri or not str(uri).startswith(('mongodb://', 'mongodb+srv://')):
        return None
    return str(uri), str(database)


@st.cache_resource(show_spinner=False)
def _client(uri):
    from pymongo import MongoClient
    return MongoClient(
        uri, serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000, socketTimeoutMS=25000,
        maxPoolSize=10, retryWrites=True
    )


def get_db():
    cfg = atlas_config()
    if cfg is None:
        raise AtlasNotConfigured('Configura [mongodb] en .streamlit/secrets.toml.')
    uri, database = cfg
    return _client(uri)[database]


def check_connection():
    get_db().client.admin.command('ping')
    return True


@st.cache_data(ttl=300, show_spinner=False)
def get_model_record(version=None):
    """Escoge modelo activo; si no hay, usa la versión V1 o la más reciente.

    No confunde la etiqueta 'evaluado' con un estado 'activo'.
    """
    modelos = get_db()['modelos']
    if version:
        return modelos.find_one({'version': version})
    return (
        modelos.find_one({'estado': 'activo'}, sort=[('fecha_registro', -1)])
        or modelos.find_one({'version': 'v1'})
        or modelos.find_one(sort=[('fecha_registro', -1)])
    )


def _gridfs_read(file_id, max_bytes):
    from gridfs import GridFS
    stream = GridFS(get_db()).get(file_id)
    if stream.length > max_bytes:
        raise ValueError('El archivo de GridFS excede el tamaño permitido.')
    return stream.read()


@st.cache_resource(show_spinner='Cargando modelo desde MongoDB Atlas…')
def _load_keras_model(gridfs_id: str, expected_sha256: str):
    """Descarga el archivo solo una vez por proceso y conserva el modelo en RAM.

    Es indispensable que el documento `modelos` sea de una fuente confiable.
    """
    import tensorflow as tf
    from bson import ObjectId
    # La clave de caché debe ser un string estable; GridFS necesita un ObjectId.
    model_bytes = _gridfs_read(ObjectId(gridfs_id), 300 * 1024 * 1024)
    actual_sha = hashlib.sha256(model_bytes).hexdigest()
    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        raise ValueError('El hash del modelo no coincide con el registrado en Atlas.')
    with tempfile.TemporaryDirectory(prefix='ecovision_model_') as tmp:
        path = Path(tmp) / 'modelo.keras'
        path.write_bytes(model_bytes)
        model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if len(model.output_shape) != 2 or int(model.output_shape[-1]) != 5:
        raise ValueError('El modelo no tiene una salida compatible con cinco clases.')
    return model


def predict_real(image: Image.Image, model_record: dict):
    """Predice sobre imagen PIL RGB y respeta el orden real de clases de Colab."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    expected_pre = model_record.get('preprocesamiento')
    if expected_pre and expected_pre != 'mobilenet_v2.preprocess_input':
        raise ValueError('El modelo requiere un preprocesamiento no admitido.')
    classes = model_record.get('categorias')
    if not isinstance(classes, list) or len(classes) != 5 or len(set(classes)) != 5:
        raise ValueError('El modelo carece de una lista válida de cinco categorías.')
    from .core import CATEGORIES
    if set(classes) != set(CATEGORIES):
        raise ValueError('Las categorías del modelo no coinciden con las del proyecto.')
    size = model_record.get('img_size', [224, 224])
    if list(size) != [224, 224]:
        raise ValueError('El tamaño de entrada del modelo no es 224 × 224.')
    model = _load_keras_model(
        str(model_record['gridfs_id']),
        model_record.get('sha256', '')
    )
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)[None, ...]
    array = preprocess_input(array)
    # Para una sola imagen, llamar directamente al modelo evita el overhead de predict().
    probabilities_raw = np.asarray(model(array, training=False).numpy()[0], dtype=float)
    if len(probabilities_raw) != 5 or not np.all(np.isfinite(probabilities_raw)):
        raise ValueError('El modelo devolvió probabilidades inválidas.')
    if np.any(probabilities_raw < 0) or probabilities_raw.sum() <= 0:
        raise ValueError('El modelo devolvió una distribución inválida.')
    probabilities_raw /= probabilities_raw.sum()
    idx = int(np.argmax(probabilities_raw))
    probabilities = {name: round(float(p) * 100, 4)
                     for name, p in zip(classes, probabilities_raw)}
    return {
        'category': classes[idx],
        'confidence': float(probabilities_raw[idx] * 100),
        'probabilities': probabilities,
        'mode': 'Real',
        'model_version': model_record.get('version', ''),
    }


def _prepare_classifier_image(image: Image.Image):
    """Crea una copia JPEG optimizada y sin EXIF; no altera la imagen en pantalla.

    El archivo conservado es una copia visual, no el JPG/PNG binario original.
    Limitamos la resolución para reducir el uso de espacio del clúster Atlas Free.
    """
    if not isinstance(image, Image.Image):
        raise ValueError('No se recibió una fotografía PIL válida.')
    photo = image.convert('RGB')
    photo.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    photo.save(output, format='JPEG', quality=82, optimize=True)
    content = output.getvalue()
    if len(content) > 8 * 1024 * 1024:
        raise ValueError('La fotografía preparada supera el límite de 8 MB para el historial.')
    return content, photo.size


def save_prediction(prediction, origin, image_digest, image=None):
    """Guarda la clasificación y, si el usuario eligió hacerlo, su fotografía.

    Compatibilidad: al omitir image, se guarda solo la clasificación como antes.
    El JPEG optimizado queda en GridFS (fs.files y fs.chunks); la clasificación
    almacena la referencia ObjectId en imagen_file_id.
    """
    from gridfs import GridFS

    db = get_db()
    doc = {
        'fecha_utc': datetime.now(timezone.utc),
        'categoria': prediction['category'],
        'confianza': float(prediction['confidence']),
        'probabilidades': {k: float(v) for k, v in prediction['probabilities'].items()},
        'origen': str(origin)[:100],
        'tipo': 'Real',
        'modelo_version': prediction['model_version'],
        'imagen_sha256': image_digest,
    }
    image_id = None
    fs = None
    if image is not None:
        content, (width, height) = _prepare_classifier_image(image)
        fs = GridFS(db)
        image_id = fs.put(
            content,
            filename=f'clasificacion_{image_digest[:16]}.jpg',
            metadata={'uso': 'imagen_clasificacion', 'imagen_sha256': image_digest},
        )
        doc.update({
            'imagen_file_id': image_id,
            'imagen_mime': 'image/jpeg',
            'imagen_ancho': width,
            'imagen_alto': height,
            'imagen_bytes': len(content),
        })
    try:
        record_id = db['clasificaciones'].insert_one(doc).inserted_id
    except Exception:
        # No dejar un archivo huérfano si falla la inserción del registro.
        if image_id is not None and fs is not None:
            try:
                fs.delete(image_id)
            except Exception:
                pass
        raise
    return str(record_id)


def get_prediction_image(record_id):
    """Recupera el JPG de una clasificación real que tenga imagen asociada.

    Los registros anteriores no contienen imagen_file_id: retorna None.
    """
    from bson import ObjectId

    if not ObjectId.is_valid(str(record_id)):
        return None
    db = get_db()
    doc = db['clasificaciones'].find_one(
        {'_id': ObjectId(str(record_id)), 'tipo': 'Real'},
        {'imagen_file_id': 1},
    )
    if not doc or not doc.get('imagen_file_id'):
        return None
    return _gridfs_read(doc['imagen_file_id'], 8 * 1024 * 1024)


def prediction_frame(limit=5000):
    """Lee resultados reales para Dashboard e Historial.

    La interfaz indica un máximo de registros para evitar consultas ilimitadas.
    """
    cur = (get_db()['clasificaciones']
           .find({'tipo': 'Real'}, {'fecha_utc': 1, 'categoria': 1, 'confianza': 1,
                                   'origen': 1, 'tipo': 1, 'imagen_file_id': 1})
           .sort('fecha_utc', -1).limit(limit))
    rows = []
    for item in cur:
        ts = item.get('fecha_utc')
        if ts is None:
            continue
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        ts = ts.tz_convert('America/Lima').tz_localize(None)
        rows.append({
            'id': str(item['_id']), 'timestamp': ts,
            'category': item.get('categoria'),
            'confidence': float(item.get('confianza', 0)),
            'origin': item.get('origen', ''), 'mode': 'Real',
            'has_image': bool(item.get('imagen_file_id')),
        })
    return pd.DataFrame(rows, columns=['id', 'timestamp', 'category', 'confidence',
                                       'origin', 'mode', 'has_image'])


def dataset_summary():
    rows = list(get_db()['imagenes'].aggregate([
        {'$group': {'_id': {'categoria': '$categoria', 'subconjunto': '$subconjunto'},
                    'cantidad': {'$sum': 1}}},
        {'$sort': {'_id.categoria': 1}}
    ]))
    return rows
