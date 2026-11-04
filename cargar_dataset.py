"""EcoVision: revisar por defecto; subir solamente con --subir. Python >=3.10."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from PIL import Image, ImageOps

CATEGORIAS = {'paper': 'Papel', 'cardboard': 'Cartón', 'plastic': 'Plástico',
              'glass': 'Vidrio', 'metal': 'Metal'}
VERSION = 'garbage-v2-5clases-v1'
FUENTE = 'https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2'


def revisar(raiz):
    faltantes = [x for x in CATEGORIAS if not (raiz / x).is_dir()]
    if faltantes:
        raise ValueError('Faltan carpetas: ' + ', '.join(faltantes))
    registros, descartados, vistos, conflictos = [], [], {}, []
    for carpeta, categoria in CATEGORIAS.items():
        for ruta in sorted((raiz / carpeta).rglob('*')):
            if not ruta.is_file():
                continue
            relativo = ruta.relative_to(raiz).as_posix()
            try:
                if ruta.is_symlink():
                    raise ValueError('Enlace simbólico no permitido')
                raw = ruta.read_bytes()
                with Image.open(io.BytesIO(raw)) as im:
                    if im.format not in ('JPEG', 'PNG'):
                        raise ValueError('Formato diferente de JPEG/PNG')
                    formato = im.format
                    im.load()
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    ancho, alto = rgb.size
                    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                sha = hashlib.sha256(raw).hexdigest()
            except Exception as exc:
                descartados.append({'ruta': relativo, 'motivo': type(exc).__name__})
                continue
            if pixels in vistos:
                previo = vistos[pixels]
                if previo['categoria'] != categoria:
                    conflictos.append([previo['ruta'], relativo])
                descartados.append({'ruta': relativo, 'motivo': 'duplicado_visual_exacto'})
                continue
            doc = {'ruta': relativo, 'categoria': categoria, 'carpeta_origen': carpeta,
                   'sha256': sha, 'sha256_pixeles': pixels, 'bytes': len(raw),
                   'ancho': ancho, 'alto': alto, 'formato': formato}
            vistos[pixels] = doc
            registros.append(doc)
    return registros, descartados, conflictos


def subir(args, registros):
    import tomli
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError
    from gridfs import GridFSBucket
    with args.secrets.open('rb') as archivo:
        cfg = tomli.load(archivo)['mongodb']
    seleccion = registros[:args.limite] if args.limite else registros
    # Claves estables permiten continuar una carga interrumpida.
    with MongoClient(cfg['uri'], serverSelectionTimeoutMS=15000) as cliente:
        cliente.admin.command('ping')
        db = cliente[cfg['database']]
        bucket = GridFSBucket(db)
        lock_id = 'ecovision_importador_dataset'
        try:
            db['importaciones_control'].insert_one({
                '_id': lock_id, 'inicio': datetime.now(timezone.utc)})
        except DuplicateKeyError:
            raise ValueError('Existe otra importación o un bloqueo de una ejecución interrumpida. '
                             'Consulta LEEME antes de reintentar.') from None
        try:
            db.imagenes.create_index([('categoria', 1), ('dataset_version', 1)])
            nuevos = omitidos = 0
            for n, doc in enumerate(seleccion, 1):
                file_id = 'ecovision:' + doc['sha256']
                registro_id = VERSION + ':' + doc['sha256_pixeles']
                existente = db.imagenes.find_one({'_id': registro_id})
                if existente and existente['categoria'] != doc['categoria']:
                    raise ValueError('Categoría en conflicto: ' + doc['ruta'])
                archivo = db['fs.files'].find_one({'_id': file_id})
                if existente and archivo:
                    omitidos += 1
                    continue
                if not archivo:
                    raw = (args.dataset / doc['ruta']).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != doc['sha256']:
                        raise ValueError('El archivo cambió después de revisarlo: ' + doc['ruta'])
                    # Solo limpia fragmentos propios de una carga incompleta; conserva archivos completos.
                    db['fs.chunks'].delete_many({'files_id': file_id})
                    bucket.upload_from_stream_with_id(
                        file_id, Path(doc['ruta']).name, io.BytesIO(raw),
                        metadata={'sha256': doc['sha256'], 'origen': 'ecovision_importador',
                                  'contentType': 'image/jpeg' if doc['formato'] == 'JPEG' else 'image/png'})
                metadatos = dict(doc, gridfs_id=file_id, dataset_version=VERSION,
                                 fuente=FUENTE, grupo_objeto=None, subconjunto='pendiente',
                                 estado_revision='pendiente', creado_en=datetime.now(timezone.utc))
                db.imagenes.update_one({'_id': registro_id}, {'$setOnInsert': metadatos}, upsert=True)
                nuevos += 1
                if n % 100 == 0 or n == len(seleccion):
                    print(f'Procesadas {n}/{len(seleccion)}', flush=True)
            print(f'Terminó: {nuevos} registros cargados/recuperados; {omitidos} ya existentes.')
        finally:
            db['importaciones_control'].delete_one({'_id': lock_id})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path, help='Carpeta Dataset ya descomprimida')
    p.add_argument('--subir', action='store_true', help='Escribe en Atlas; por defecto solo revisa')
    p.add_argument('--limite', type=int, default=0, help='Subir hasta N imágenes; 0 = todas')
    p.add_argument('--secrets', type=Path, default=Path('.streamlit/secrets.toml'))
    p.add_argument('--salida', type=Path, default=Path('revision_dataset'))
    args = p.parse_args()
    if args.limite < 0:
        p.error('--limite debe ser mayor o igual a 0')
    args.dataset = args.dataset.resolve()
    registros, descartados, conflictos = revisar(args.dataset)
    counts = Counter(x['categoria'] for x in registros)
    total_bytes = sum(x['bytes'] for x in registros)
    resumen = {'validas_unicas': len(registros), 'por_categoria': dict(counts),
               'bytes_imagenes': total_bytes, 'MiB_imagenes': round(total_bytes / 1024**2, 2),
               'descartados': descartados, 'conflictos_etiqueta': conflictos,
               'nota': 'Tamaño solo de imágenes. No incluye metadatos, índices ni otros datos del clúster.'}
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / 'resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.salida / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for doc in registros:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    if conflictos or any(counts[c] == 0 for c in CATEGORIAS.values()):
        raise ValueError('Revisa categorías vacías o conflictos antes de cargar.')
    if args.subir:
        subir(args, registros)
    else:
        print('Revisión local terminada. No se conectó a Atlas ni se cargaron imágenes.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # No mostrar URI o credenciales incluidas en mensajes del driver.
        print('ERROR:', type(exc).__name__, file=sys.stderr)
        if isinstance(exc, (ValueError, FileNotFoundError, KeyError)):
            print(str(exc), file=sys.stderr)
        else:
            print('Revisa dependencias, acceso de red y credenciales. Puedes volver a ejecutar.', file=sys.stderr)
        sys.exit(1)
