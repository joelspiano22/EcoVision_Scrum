from __future__ import annotations
import hashlib
import io
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
import pandas as pd
from .core import CATEGORIES, SLUGS, image_hash, open_image

MAX_BYTES = 60 * 1024 * 1024

def prepare_zip(raw: bytes, destination: Path, seed=42):
    """ZIP con categoria/objeto_id/foto.jpg; agrupa vistas antes de dividir."""
    if len(raw) > MAX_BYTES:
        raise ValueError("El ZIP puede ocupar hasta 60 MB.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="dataset_", dir=destination.parent))
    rows, duplicates, seen = [], 0, {}
    synthetic = False
    try:
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise ValueError("El archivo no es un ZIP válido.") from exc
        with archive as z:
            files = z.infolist()
            if len(files) > 2000 or sum(f.file_size for f in files) > 200 * 1024 * 1024:
                raise ValueError("Máximo 2000 entradas y 200 MB descomprimidos.")
            synthetic = any(PurePosixPath(f.filename).name == "DATOS_SINTETICOS.txt" for f in files)
            for entry in files:
                p = PurePosixPath(entry.filename.replace('\\', '/'))
                if p.is_absolute() or '..' in p.parts or ':' in entry.filename:
                    raise ValueError("El ZIP contiene una ruta no permitida.")
                if entry.is_dir() or p.suffix.lower() not in {'.jpg','.jpeg','.png'}:
                    continue
                if any(part.startswith('.') or part == '__MACOSX' for part in p.parts):
                    continue
                parts = p.parts
                # Admite una carpeta contenedora; el último bloque siempre debe ser categoría/objeto/foto.
                if len(parts) < 3 or parts[-3].lower() not in SLUGS:
                    raise ValueError(f"Estructura incorrecta: {entry.filename}. Usa categoria/objeto_id/foto.jpg.")
                label = CATEGORIES[SLUGS.index(parts[-3].lower())]
                object_id = parts[-3].lower() + '/' + parts[-2]
                image = open_image(z.read(entry))
                digest = image_hash(image)
                if digest in seen:
                    if seen[digest] != label:
                        raise ValueError("Una misma imagen tiene etiquetas diferentes. Corrige el ZIP.")
                    duplicates += 1
                    continue
                seen[digest] = label
                relative = 'images/' + digest + '.png'
                target = staging / relative
                target.parent.mkdir(exist_ok=True)
                image.save(target)
                rows.append({"path":relative, "category":label, "object_id":object_id, "sha256":digest, "original":entry.filename})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("No se encontraron fotografías JPG o PNG.")
        assignments = {}
        for category in CATEGORIES:
            groups = sorted(frame.loc[frame.category == category, 'object_id'].unique())
            if len(groups) < 5:
                raise ValueError(f"{category}: necesitas al menos 5 objetos diferentes, cada uno en su carpeta.")
            random.Random(seed + CATEGORIES.index(category)).shuffle(groups)
            n_test = max(1, int(len(groups)*.2))
            n_val = max(1, int(len(groups)*.2))
            for i, group in enumerate(groups):
                assignments[group] = 'test' if i < n_test else 'validation' if i < n_test+n_val else 'train'
        frame['split'] = frame.object_id.map(assignments)
        frame.to_csv(staging/'manifest.csv', index=False)
        metadata = {"seed":seed,"categories":CATEGORIES,"images":len(frame),"objects":frame.object_id.nunique(),
                    "duplicates_removed":duplicates,"synthetic":synthetic,
                    "zip_sha256":hashlib.sha256(raw).hexdigest(),
                    "split_rule":"60/20/20 aproximado por objetos dentro de cada categoría; redondeo a enteros",
                    "limitation":"La agrupación depende de que las vistas del mismo objeto estén en la misma carpeta. No detecta automáticamente objetos iguales en carpetas distintas."}
        (staging/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
        # Cada ZIP y semilla tiene su directorio: no sobrescribe conjuntos anteriores.
        if destination.exists():
            shutil.rmtree(staging)
        else:
            staging.rename(destination)
        return pd.read_csv(destination/'manifest.csv'), json.loads((destination/'metadata.json').read_text(encoding='utf8'))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

def load_dataset(path):
    path = Path(path)
    return pd.read_csv(path/'manifest.csv'), json.loads((path/'metadata.json').read_text(encoding='utf8'))
