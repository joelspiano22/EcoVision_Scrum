"""Explorador de solo lectura del dataset cargado por cargar_dataset.py."""
import io
import math
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from .atlas_service import atlas_config, _client

CATEGORIAS = ['Papel', 'Cartón', 'Plástico', 'Vidrio', 'Metal']
VERSION = 'garbage-v2-5clases-v1'


@st.cache_resource(show_spinner=False)
def cliente_atlas(uri):
    return _client(uri)


@st.cache_data(ttl=20, max_entries=16, show_spinner=False)
def resumen_atlas(uri, database):
    db = cliente_atlas(uri)[database]
    pipeline = [{'$match': {'dataset_version': VERSION}},
                {'$group': {'_id': '$categoria', 'cantidad': {'$sum': 1},
                            'bytes': {'$sum': '$bytes'}}}]
    rows = list(db.imagenes.aggregate(pipeline, maxTimeMS=10000))
    return rows, datetime.now().strftime('%H:%M:%S')


@st.cache_data(ttl=20, max_entries=64, show_spinner=False)
def pagina_atlas(uri, database, categoria, subconjunto, pagina, cantidad):
    filtro = {'dataset_version': VERSION}
    if categoria != 'Todas':
        filtro['categoria'] = categoria
    if subconjunto != 'Todos':
        filtro['subconjunto'] = subconjunto
    campos = {'categoria': 1, 'ruta': 1, 'ancho': 1, 'alto': 1, 'bytes': 1,
              'gridfs_id': 1, 'subconjunto': 1, 'estado_revision': 1}
    cursor = cliente_atlas(uri)[database].imagenes.find(filtro, campos).sort('_id', 1)
    return list(cursor.skip((pagina - 1) * cantidad).limit(cantidad).max_time_ms(10000))


@st.cache_data(ttl=600, max_entries=96, show_spinner=False)
def miniatura_atlas(uri, database, file_id):
    from gridfs import GridFSBucket
    bucket = GridFSBucket(cliente_atlas(uri)[database])
    with bucket.open_download_stream(file_id) as stream:
        if stream.length > 15 * 1024**2:
            raise ValueError('Imagen demasiado grande para vista previa')
        raw = stream.read()
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((320, 320))
        output = io.BytesIO()
        im.save(output, format='JPEG', quality=80)
        return output.getvalue()


def render_atlas(root: Path):
    st.title('Dataset en Atlas')
    st.caption('Explora imágenes reales desde GridFS y consulta los subconjuntos train, val y test guardados en Atlas.')
    try:
        import pymongo
        from gridfs.errors import NoFile, CorruptGridFile
    except ImportError:
        st.error('Instala las dependencias: python -m pip install -r requirements-atlas.txt')
        return
    config = atlas_config()
    if config is None:
        st.info('Configura [mongodb] en .streamlit/secrets.toml o MONGO_URI y MONGO_DB en el entorno.')
        return
    uri, database = config
    if st.button('Actualizar datos', key='atlas_refresh'):
        resumen_atlas.clear()
        pagina_atlas.clear()
    try:
        rows, hora = resumen_atlas(uri, database)
        por_categoria = {r['_id']: r['cantidad'] for r in rows}
        total = sum(r['cantidad'] for r in rows)
        size = sum(r['bytes'] for r in rows)
        a, b, c = st.columns(3)
        a.metric('Imágenes registradas', f'{total:,}')
        b.metric('Referencia de carga', '7010')
        c.metric('Imágenes almacenadas', f'{size / 1024**2:.2f} MiB')
        st.progress(min(total / 7010, 1.0))
        st.caption(f'Consulta: {hora} (hora del servidor). Referencia: 7010 imágenes del dataset revisado. '
                   'El tamaño no incluye índices ni otros datos de Atlas.')
        if total < 7010:
            st.info('La carga todavía está incompleta respecto a la referencia. Pulsa Actualizar datos para consultar el avance.')
        else:
            st.success('La cantidad de imágenes coincide con la referencia de 7,010. Se visualiza el dataset cargado en Atlas.')
        frame = pd.DataFrame({'Categoría': CATEGORIAS,
                              'Imágenes': [por_categoria.get(c, 0) for c in CATEGORIAS]})
        st.bar_chart(frame.set_index('Categoría'))
        st.dataframe(frame, hide_index=True, width='stretch')
        if not total:
            st.info('Aún no hay imágenes de esta versión del dataset. Mantén el importador ejecutándose y actualiza después.')
            return
        c1, c2, c3 = st.columns(3)
        categoria = c1.selectbox('Categoría', ['Todas'] + CATEGORIAS, key='atlas_categoria')
        subconjunto = c2.selectbox('Subconjunto', ['Todos', 'train', 'val', 'test', 'pendiente'], key='atlas_split')
        cantidad = c3.selectbox('Imágenes por página', [12, 24], key='atlas_cantidad')
        filtro = {'dataset_version': VERSION}
        if categoria != 'Todas':
            filtro['categoria'] = categoria
        if subconjunto != 'Todos':
            filtro['subconjunto'] = subconjunto
        count = cliente_atlas(uri)[database].imagenes.count_documents(filtro)
        if count == 0:
            st.info('Esta categoría aún no tiene imágenes registradas.')
            return
        pages = max(1, math.ceil(count / cantidad))
        page_key = f'atlas_page_{categoria}_{subconjunto}_{cantidad}'
        if st.session_state.get(page_key, 1) > pages:
            st.session_state[page_key] = 1
        pagina = int(st.number_input('Página', 1, pages, value=1, step=1, key=page_key))
        st.caption(f'Página {pagina} de {pages}. Durante la carga, la posición de las imágenes puede cambiar al actualizar.')
        docs = pagina_atlas(uri, database, categoria, subconjunto, pagina, cantidad)
        if not docs:
            st.info('No hay registros en esta página. Actualiza los datos.')
            return
        show = st.checkbox('Mostrar fotografías', value=False, key='atlas_fotos',
                           help='Descarga solo las imágenes de la página visible. Puedes dejarlo apagado mientras cargas el dataset.')
        if show:
            columns = st.columns(3)
            for index, doc in enumerate(docs):
                with columns[index % 3]:
                    with st.container(border=True):
                        try:
                            preview = miniatura_atlas(uri, database, doc['gridfs_id'])
                            st.image(preview, width='stretch')
                        except (NoFile, CorruptGridFile, OSError, ValueError, KeyError):
                            st.warning('Vista previa no disponible para esta imagen.')
                        st.write(doc.get('categoria', 'Sin categoría'))
                        st.caption(doc.get('ruta', ''))
        visible = pd.DataFrame([{
            'Categoría': d.get('categoria'), 'Archivo': d.get('ruta'),
            'Ancho': d.get('ancho'), 'Alto': d.get('alto'),
            'KiB': round(d.get('bytes', 0) / 1024, 2),
            'Subconjunto': d.get('subconjunto', 'pendiente'),
            'Revisión': d.get('estado_revision', 'pendiente'),
        } for d in docs])
        st.dataframe(visible, hide_index=True, width='stretch')
        st.download_button('Descargar metadatos de esta página',
                           visible.to_csv(index=False).encode('utf-8-sig'),
                           'dataset_pagina.csv', 'text/csv')
        st.caption('Las categorías y subconjuntos provienen de Atlas; no son predicciones. Las fotografías se recuperan desde GridFS solo al activar la vista previa.')
    except pymongo.errors.PyMongoError as exc:
        st.error('No se pudo consultar Atlas. Revisa la conexión, el usuario y los permisos; luego pulsa Actualizar datos.')
        st.caption('Tipo de error: ' + type(exc).__name__)
