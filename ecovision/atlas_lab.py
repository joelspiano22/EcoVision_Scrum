"""Laboratorio IA conectado a los modelos y métricas de MongoDB Atlas."""
from __future__ import annotations
import io
import json
import pandas as pd
import streamlit as st
from .atlas_service import get_db, get_model_record, predict_real, dataset_summary
from .core import CATEGORIES, open_image


def render_atlas_lab():
    st.markdown('### Modelos entrenados en MongoDB Atlas')
    st.caption('Consulta modelos y métricas almacenados por Google Colab. El entrenamiento se ejecuta fuera de Streamlit.')
    db = get_db()
    registros = list(db['modelos'].find({}, {'nombre': 1, 'version': 1, 'arquitectura': 1,
          'estado': 1, 'metricas': 1, 'gridfs_id': 1, 'fecha_registro': 1,
          'sha256': 1, 'categorias': 1, 'img_size': 1, 'preprocesamiento': 1,
          'dataset_version': 1}).sort('fecha_registro', -1).limit(50))
    t1,t2,t3 = st.tabs(['01 · Modelos en Atlas', '02 · Dataset de entrenamiento', '03 · Probar predicción'])
    if not registros:
        st.info('No hay modelos registrados en la colección modelos. Comprueba el paso de guardado en Colab.')
        return
    opciones = {f"{r.get('nombre', 'Modelo')} · {r.get('version', 'sin versión')}": r for r in registros}
    with t1:
        rows = []
        for r in registros:
            metrics = r.get('metricas') or {}
            rows.append({'Versión': r.get('version'), 'Arquitectura': r.get('arquitectura'),
                         'Estado': r.get('estado'), 'Accuracy test (%)': round(metrics.get('accuracy_test', 0) * 100, 2)
                            if 'accuracy_test' in metrics else None,
                         'F1 macro test (%)': round(metrics.get('f1_macro_test', 0) * 100, 2)
                            if 'f1_macro_test' in metrics else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        selected = st.selectbox('Detalles del modelo', list(opciones), key='lab_model_info')
        record = opciones[selected]
        metrics = record.get('metricas') or {}
        c1,c2,c3 = st.columns(3)
        c1.metric('Exactitud prueba', f"{metrics['accuracy_test'] * 100:.2f}%" if 'accuracy_test' in metrics else 'No registrada')
        c2.metric('F1 macro prueba', f"{metrics['f1_macro_test'] * 100:.2f}%" if 'f1_macro_test' in metrics else 'No registrado')
        c3.metric('Imágenes de prueba', str(metrics.get('imagenes_test', 'No registrado')))
        st.write('**Versión:**', record.get('version', '-'))
        st.write('**Estado:**', record.get('estado', '-'))
        st.write('**Dataset:**', record.get('dataset_version', '-'))
        st.write('**Categorías (orden de salida):**', ', '.join(record.get('categorias') or []))
        if not record.get('categorias'):
            st.warning('Falta el orden de categorías: es necesario para inferencia fiable.')
        st.caption('La matriz de confusión, las curvas por época y los valores por clase aparecerán cuando se registren en la colección modelos. No se reconstruyen a partir de métricas generales.')
    with t2:
        rows = dataset_summary()
        data = [{'Categoría': r['_id'].get('categoria', '-'),
                 'Subconjunto': r['_id'].get('subconjunto', 'pendiente'),
                 'Imágenes': r['cantidad']} for r in rows]
        if data:
            df = pd.DataFrame(data)
            st.metric('Imágenes registradas en Atlas', f"{int(df['Imágenes'].sum()):,}")
            crosstab = df.pivot_table(index='Categoría', columns='Subconjunto',
                                     values='Imágenes', aggfunc='sum', fill_value=0)
            st.dataframe(crosstab.reindex(CATEGORIES).fillna(0).astype(int), width='stretch')
            st.download_button('Descargar distribución CSV', df.to_csv(index=False).encode('utf-8-sig'),
                               'ecovision_distribucion_atlas.csv', 'text/csv')
        else:
            st.info('No existen registros del dataset.')
    with t3:
        activo = get_model_record()
        if not activo:
            st.warning('No hay un modelo activo para realizar predicciones.')
            return
        st.caption(f"Modelo seleccionado automáticamente: {activo.get('version', 'sin versión')}")
        uploaded = st.file_uploader('Selecciona fotografía de prueba', type=['jpg', 'jpeg', 'png'], key='atlas_lab_file')
        if st.button('Predecir con modelo de Atlas', disabled=uploaded is None, key='atlas_lab_predict'):
            try:
                image = open_image(uploaded.getvalue())
                result = predict_real(image, activo)
                st.image(image, width=250)
                st.success(f"{result['category']} · {result['confidence']:.1f}%")
                st.dataframe(pd.DataFrame({'Material': result['probabilities'].keys(),
                             'Probabilidad (%)': result['probabilities'].values()}),
                             hide_index=True)
                st.caption('Prueba técnica: no crea registro en el historial.')
            except ImportError:
                st.error('TensorFlow no está instalado. Ejecuta: pip install -r requirements-cnn.txt')
            except Exception as exc:
                st.error('No se pudo realizar la inferencia. Verifica el modelo registrado y GridFS.')
                st.caption(f'Tipo de error: {type(exc).__name__}')
