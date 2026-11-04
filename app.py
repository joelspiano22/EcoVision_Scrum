from __future__ import annotations
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw
from ecovision.core import (CATEGORIES, COLORS, GUIDANCE, Store, demo_prediction,
    filter_frame, image_hash, open_image, visible_frame)
from ecovision.dataset import prepare_zip, load_dataset
from ecovision.training import train_model, predict_saved
from ecovision.atlas_service import (atlas_config, get_model_record, predict_real,
                                    save_prediction, prediction_frame, check_connection,
                                    get_prediction_image)

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('ECOVISION_DATA_DIR',ROOT/'data'))
DATA.mkdir(parents=True,exist_ok=True)
st.set_page_config(page_title='EcoVision · Reciclar con criterio',page_icon='♻️',layout='wide')
USE_ATLAS = atlas_config() is not None
store = None if USE_ATLAS else Store(DATA/'ecovision.sqlite3')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {--mint:#57e6c0;--ink:#080e1b;--muted:#95a8c5;}
html,body,[class*="st-"] {font-family:'Space Grotesk',sans-serif;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#123335 0%,transparent 44%),#080e1b;}
.block-container {padding-top:2rem;max-width:1440px;}
[data-testid="stSidebar"] {background:#0b1424;border-right:1px solid #203149;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#13243b,#101c2d);border:1px solid #293a51;border-radius:16px;padding:18px;}
[data-testid="stMetricLabel"] {color:#a9bdd6;}
[data-testid="stMetricValue"] {color:#edf8ff;}
h1,h2,h3 {letter-spacing:-.04em;}
.eyebrow {color:#57e6c0;font-size:.76rem;letter-spacing:.18em;font-weight:700;margin-bottom:12px;}
.hero {padding:30px 32px;border:1px solid #294252;border-radius:24px;background:linear-gradient(120deg,#132b39dd,#0d182aee);margin:8px 0 24px;position:relative;overflow:hidden;}
.hero h1 {font-size:2.9rem;line-height:1.08;margin:0 0 12px;max-width:850px;}
.hero p {color:#a9bdd6;max-width:760px;font-size:1rem;margin:0;}
.chip {display:inline-block;border:1px solid #355163;border-radius:30px;padding:5px 12px;color:#8cf1d5;font-size:.76rem;margin-right:6px;margin-top:18px;}
.brand {font-weight:700;font-size:1.9rem;letter-spacing:-.06em;color:#f0faff;}
.brand span {color:#57e6c0;}
.muted {color:#95a8c5;font-size:.85rem;}
.result {border:1px solid #356e67;border-radius:20px;padding:22px;background:linear-gradient(130deg,#153a37,#122235);margin-bottom:16px;}
.result .material {font-size:2.4rem;font-weight:700;line-height:1.3;}
.result .score {color:#79f5d1;font-size:1.2rem;}
.step {color:#57e6c0;font-size:.78rem;letter-spacing:.12em;}
.footer {border-top:1px solid #213149;margin-top:32px;padding-top:16px;color:#8397b3;font-size:.8rem;}
@media(max-width:700px){.hero {padding:22px 18px}.hero h1{font-size:2rem}.block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}}
</style>''',unsafe_allow_html=True)

st.session_state.setdefault('upload_version',0)
st.session_state.setdefault('result',None)
st.session_state.setdefault('sample',None)
st.session_state.setdefault('sample_name',None)

def hero(kicker,title,description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p><span class="chip">I. E. 6064 Francisco Bolognesi</span><span class="chip">Prototipo académico</span></div>',unsafe_allow_html=True)

def reset_input():
    st.session_state.upload_version+=1
    st.session_state.result=None
    st.session_state.sample=None
    st.session_state.sample_name=None

def mostrar_error_atlas(etapa, exc):
    """Diagnóstico útil sin mostrar la URI ni la contraseña de MongoDB."""
    tipo = type(exc).__name__
    codigo = getattr(exc, 'code', None)
    if tipo == 'OperationFailure' and codigo == 13:
        mensaje = ('MongoDB rechazó la escritura (código 13: sin autorización). '
                   'En Atlas > Database Access revisa que el usuario del secrets.toml '
                   'tenga readWrite sobre ecovision.')
    elif tipo == 'OperationFailure' and codigo == 18:
        mensaje = 'Error de autenticación en Atlas. Revisa el usuario y la contraseña en Secrets.'
    elif tipo in {'ServerSelectionTimeoutError', 'NetworkTimeout', 'AutoReconnect',
                  'ConnectionFailure', 'WTimeoutError'}:
        mensaje = ('La conexión con MongoDB Atlas falló o agotó el tiempo. '
                   'Revisa la conectividad, la lista de IP permitidas y el estado del clúster.')
    elif tipo == 'GridFSError' or tipo == 'NoFile':
        mensaje = 'No se pudo leer el archivo del modelo desde GridFS.'
    elif tipo in {'ImportError', 'ModuleNotFoundError'}:
        mensaje = 'Falta una dependencia de IA (TensorFlow/Keras). Revisa requirements-cnn.txt.'
    elif isinstance(exc, ValueError):
        # Los mensajes ValueError provienen de validaciones de la propia app o Keras.
        # Evitar mostrar rutas o posibles credenciales en errores externos.
        mensaje = ('El modelo o sus metadatos no tienen el formato esperado. '
                   'Revisa la versión de TensorFlow y los datos de modelos en Atlas.')
    else:
        mensaje = 'La operación no pudo completarse. Revisa los detalles de diagnóstico.'
    st.error(f'{etapa}: {mensaje}')
    st.caption('Diagnóstico: ' + tipo + (f' (código MongoDB {codigo})' if codigo is not None else ''))


def example_image():
    image=Image.new('RGB',(640,480),'#e7eff4')
    d=ImageDraw.Draw(image)
    d.rounded_rectangle((245,95,395,400),radius=38,fill='#97d7d7',outline='#277d84',width=5)
    d.rounded_rectangle((277,60,363,109),radius=8,fill='#2b7181')
    d.rectangle((244,230,396,310),fill='#ecfbf7')
    d.text((275,258),'ECO DEMO',fill='#174b55')
    raw=io.BytesIO();image.save(raw,format='PNG')
    return raw.getvalue()

with st.sidebar:
    st.markdown('<div class="brand">eco<span>vision</span> ◉</div><p class="muted">Inteligencia para separar mejor</p>',unsafe_allow_html=True)
    st.divider()
    page=st.radio('Espacio de trabajo',['Clasificador','Dashboard','Historial','Dataset en Atlas','Laboratorio IA','Validación de sprints'],label_visibility='collapsed')
    st.divider()
    st.markdown('**FRONTEND · ' + ('MONGODB ATLAS' if USE_ATLAS else 'MODO DEMOSTRACIÓN') + '**')
    if USE_ATLAS:
        st.caption('El clasificador utiliza el modelo real registrado en Atlas. Los resultados se guardan en clasificaciones y las fotografías autorizadas en GridFS.')
    else:
        st.caption('No se ha configurado Atlas; esta sesión utiliza datos simulados y SQLite local.')
    threshold=st.slider('Umbral de advertencia (%)',0,100,70,help='Si la confianza queda debajo del umbral se muestra una advertencia; no es una garantía de acierto.')
    st.caption('Cinco materiales · Un residuo por fotografía')
    with st.expander('Alcance y cuidado de las imágenes'):
        st.write('No admite residuos peligrosos, sanitarios ni electrónicos. Evita rostros, nombres y otros datos personales. La persona mantiene la decisión final.')
        st.write('Al seleccionar «Guardar fotografía en Atlas», se almacenará una copia JPEG optimizada y sin EXIF en GridFS. Si no lo seleccionas, solo se registra la predicción y su hash. No subas imágenes con personas ni datos personales. Las imágenes importadas al laboratorio mantienen sus reglas anteriores.')

if page=='Clasificador':
    hero('01 / EXPLORA · CAPTURA · COMPRENDE','Una imagen. Una mejor decisión.',
         'Carga una fotografía para clasificar su material con el modelo seleccionado desde MongoDB Atlas.' if USE_ATLAS
         else 'Carga una fotografía y prueba la interfaz en modo demostración.')
    if USE_ATLAS:
        st.info('PREDICCIÓN REAL: se utilizará el modelo guardado en Atlas. La confianza no garantiza que la categoría sea correcta.')
    else:
        st.info('SIMULACIÓN: la categoría y los porcentajes se generan para probar la interfaz. No identifican el material real de la fotografía.')
    left,right=st.columns([1.05,1],gap='large')
    with left:
        st.markdown('<div class="step">01 · SELECCIONA UNA FOTOGRAFÍA</div>',unsafe_allow_html=True)
        source=st.radio('Origen',['Cargar archivo','Usar cámara'],horizontal=True)
        guardar_foto = False
        if USE_ATLAS:
            guardar_foto = st.checkbox(
                'Guardar fotografía en MongoDB Atlas para verla después en Historial',
                value=False,
                help='Opcional: se almacena una copia JPG optimizada sin metadatos EXIF. '
                     'Desmarca esta opción para guardar únicamente el resultado.',
            )
            st.caption('Por privacidad, el guardado de fotografías es opcional. '
                       'Evita subir imágenes con rostros o información personal.')
        if source=='Cargar archivo':
            file=st.file_uploader('JPG o PNG · Un residuo · Hasta 15 MB',type=['jpg','jpeg','png'],key=f'upload_{st.session_state.upload_version}')
        else:
            file=st.camera_input('Permite el acceso a la cámara y toma una fotografía',key=f'camera_{st.session_state.upload_version}')
            st.caption('Si la cámara no está disponible o se deniega el permiso, selecciona «Cargar archivo». En un servidor remoto, el navegador requiere un contexto seguro (HTTPS).')
        if st.button('Probar con imagen de ejemplo',width='stretch'):
            reset_input()
            st.session_state.sample=example_image()
            st.session_state.sample_name='Ilustración de ejemplo'
            st.rerun()
        raw=file.getvalue() if file is not None else st.session_state.sample
        origin=source if file is not None else st.session_state.sample_name
        image=None;digest=None
        if raw:
            try:
                image=open_image(raw);digest=image_hash(image)
                if st.session_state.result and st.session_state.result['digest']!=digest:
                    st.session_state.result=None
                st.image(image,caption=f'{origin} · {image.width} × {image.height} px',width='stretch')
            except ValueError as exc:
                st.session_state.result=None
                st.error(str(exc))
        else:
            st.session_state.result=None
            st.markdown('''<div style="border:1px dashed #3a536e;border-radius:18px;padding:50px 25px;text-align:center;color:#95a8c5"><div style="font-size:3rem;color:#57e6c0">◎</div>Tu próxima consulta comienza aquí.<br>Selecciona un archivo o utiliza el ejemplo.</div>''',unsafe_allow_html=True)
        a,b=st.columns([2,1])
        with a:
            analyze=st.button('Analizar imagen →',type='primary',disabled=image is None,width='stretch')
        with b:
            st.button('Limpiar',on_click=reset_input,width='stretch')
        if analyze and image:
            if USE_ATLAS:
                model_info = None
                try:
                    with st.spinner('1/3 · Consultando el modelo en MongoDB Atlas…'):
                        model_info = get_model_record()
                except Exception as exc:
                    mostrar_error_atlas('No se pudo consultar la colección modelos', exc)
                if model_info is None:
                    if not st.session_state.result:
                        st.warning('No se encontró un modelo disponible en Atlas o no fue posible consultarlo.')
                else:
                    prediction = None
                    start_prediction = time.perf_counter()
                    try:
                        with st.spinner('2/3 · Cargando MobileNetV2 y clasificando…'):
                            prediction = predict_real(image, model_info)
                    except Exception as exc:
                        mostrar_error_atlas('No se pudo clasificar la imagen', exc)
                    if prediction is not None:
                        duration = time.perf_counter() - start_prediction
                        # La predicción correcta se muestra aunque falle el registro del historial.
                        st.session_state.result = {
                            **prediction, 'id': 'no-guardado', 'digest': digest,
                            'saved': False,
                        }
                        st.caption(f'Inferencia: {duration:.1f} s. La primera ejecución '
                                   'puede incluir la descarga y carga del modelo desde GridFS.')
                        try:
                            with st.spinner('3/3 · Guardando la clasificación en Atlas…'):
                                record_id = save_prediction(
                                    prediction, origin or 'Archivo', digest,
                                    image=image if guardar_foto else None,
                                )
                            st.session_state.result['id'] = record_id
                            st.session_state.result['saved'] = True
                            if guardar_foto:
                                st.success('Clasificación y fotografía guardadas en MongoDB Atlas.')
                            else:
                                st.success('Clasificación registrada en Atlas (sin fotografía).')
                        except Exception as exc:
                            mostrar_error_atlas('La predicción se calculó, pero no pudo guardarse en Historial', exc)
            else:
                try:
                    with st.spinner('Preparando demostración…'):
                        prediction = demo_prediction(image)
                        record_id = store.add(prediction, origin or 'Archivo', digest)
                        st.session_state.result = {**prediction, 'id': record_id, 'digest': digest}
                except Exception as exc:
                    st.error('No se pudo generar la demostración.')
                    st.caption('Diagnóstico: ' + type(exc).__name__)
        if USE_ATLAS and st.session_state.sample is not None:
            st.caption('La ilustración de ejemplo es sintética: la predicción no permite evaluar el rendimiento real del modelo.')
        st.caption('Un clic en analizar crea un registro. Limpiar elimina la consulta de pantalla; el historial se conserva.')
    with right:
        st.markdown('<div class="step">02 · INTERPRETA EL RESULTADO</div>',unsafe_allow_html=True)
        result=st.session_state.result
        if result:
            tipo_resultado = (
                "RESULTADO REAL" if result.get("mode") == "Real"
                else "RESULTADO SIMULADO"
            )
            tipo_confianza = (
                " del modelo" if result.get("mode") == "Real"
                else " simulada"
            )
            st.markdown(
                f"""<div class="result">
                    <div class="eyebrow">{tipo_resultado} · #{str(result['id'])[:8]}</div>
                    <div class="material">{result['category']}</div>
                    <div class="score">{result['confidence']:.1f}% de confianza{tipo_confianza}</div>
                </div>""",
                unsafe_allow_html=True
            )
            st.progress(min(1.,result['confidence']/100))
            if result.get('mode') == 'Real' and result.get('saved') is False:
                st.warning('Predicción realizada, pero pendiente de guardar en Atlas. No aparecerá en Historial ni Dashboard hasta corregir el acceso de escritura.')
            probs=pd.DataFrame({'Material':list(result['probabilities']),'Probabilidad (%)' if USE_ATLAS else 'Probabilidad simulada (%)':list(result['probabilities'].values())})
            fig=px.bar(probs,x='Probabilidad (%)' if USE_ATLAS else 'Probabilidad simulada (%)',y='Material',orientation='h',color='Material',color_discrete_map=COLORS)
            fig.update_layout(height=260,showlegend=False,margin=dict(l=0,r=10,t=5,b=0),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_range=[0,100])
            st.plotly_chart(fig,width='stretch')
            if result['confidence']<threshold:
                st.warning('Confianza por debajo del umbral configurado. Repite la fotografía con buena iluminación o revisa el material con una persona responsable.')
            st.markdown('**Orientación para la categoría mostrada**')
            st.write(GUIDANCE[result['category']])
            st.caption('Orientación general editable: confirmar las reglas del punto de recolección de la institución. El porcentaje no equivale a exactitud ni garantiza reciclabilidad.')
        else:
            st.markdown('### El resultado aparecerá aquí')
            st.write('Verás la categoría sugerida, la distribución de porcentajes y una orientación para revisar la separación.')
            for category in CATEGORIES:
                st.markdown(f'<span class="chip">{category}</span>',unsafe_allow_html=True)
            st.markdown('')
            st.caption('No se muestran resultados hasta que inicies un análisis.')
    with st.expander('Orientaciones disponibles para las cinco categorías'):
        for category in CATEGORIES:
            st.markdown(f'**{category}.** {GUIDANCE[category]}')

elif page in {'Dashboard','Historial'}:
    hero('02 / DATOS CON CONTEXTO','El panorama de tus consultas.' if page=='Dashboard' else 'Cada consulta, en un solo lugar.',
         'Explora los resultados reales del clasificador almacenados en MongoDB Atlas.' if USE_ATLAS
         else 'Explora las consultas simuladas registradas en SQLite local.')
    if USE_ATLAS:
        st.caption('Solo predicciones reales; hasta 5,000 registros recientes. La confianza del modelo no equivale a su exactitud.')
        if st.button('Actualizar registros', type='primary'):
            st.rerun()
    else:
        st.warning('DATOS SIMULADOS · Estos indicadores no representan aciertos del modelo ni resultados ambientales.')
    if not USE_ATLAS and st.button('Cargar 20 registros de demostración',type='primary'):
        if store.seed(): st.success('Se agregaron 20 registros simulados.')
        else: st.info('El lote de 20 registros ya estaba cargado; no se duplicó.')
        for key in ['filter_categories','filter_confidence','filter_dates']:
            st.session_state.pop(key,None)
    try:
        df = prediction_frame() if USE_ATLAS else store.frame()
    except Exception as exc:
        st.error('No se pudo consultar el historial real desde MongoDB Atlas.')
        st.caption('Tipo de error: ' + type(exc).__name__)
        df = pd.DataFrame(columns=['id','timestamp','category','confidence','origin','mode','has_image'])
    if df.empty:
        st.info('Todavía no hay predicciones reales. Realiza una clasificación para iniciar el historial.' if USE_ATLAS else 'Todavía no hay análisis. Utiliza el clasificador o carga los 20 registros de demostración.')
    else:
        with st.container(border=True):
            c1,c2,c3=st.columns([1.2,1,1.2])
            with c1:
                categories=st.multiselect('Materiales',CATEGORIES,default=CATEGORIES,key='filter_categories')
            with c2:
                confidence=st.slider('Confianza (%)' if USE_ATLAS else 'Confianza simulada (%)',0,100,(0,100),key='filter_confidence')
            with c3:
                dates=st.date_input('Periodo (hora de Perú)',value=(df.timestamp.min().date(),df.timestamp.max().date()),key='filter_dates')
            if st.button('Restablecer filtros'):
                for key in ['filter_categories','filter_confidence','filter_dates']:
                    st.session_state.pop(key,None)
                st.rerun()
        if not isinstance(dates,(list,tuple)) or len(dates)!=2:
            st.info('Selecciona la fecha de inicio y la fecha de fin.')
        else:
            filtered=filter_frame(df,categories,confidence,dates)
            st.caption(f'{len(filtered)} de {len(df)} registros · Las cifras y gráficos comparten los mismos filtros.')
            if filtered.empty:
                st.info('No hay resultados que coincidan con los filtros. Amplía la selección o restablécelos.')
            else:
                if page=='Dashboard':
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric('Predicciones reales' if USE_ATLAS else 'Análisis simulados',len(filtered))
                    c2.metric('Confianza promedio',f'{filtered.confidence.mean():.1f}%')
                    c3.metric('Categorías presentes',filtered.category.nunique())
                    c4.metric('Bajo el umbral',int((filtered.confidence<threshold).sum()))
                    counts=filtered.groupby('category').size().reindex(CATEGORIES,fill_value=0).rename('Análisis').reset_index().rename(columns={'category':'Material'})
                    l,r=st.columns([1,1.25])
                    with l:
                        st.markdown('#### Distribución por material')
                        fig=px.pie(counts,names='Material',values='Análisis',hole=.67,color='Material',color_discrete_map=COLORS)
                        fig.update_traces(textinfo='percent+label')
                        fig.update_layout(height=340,showlegend=False,margin=dict(l=20,r=20,t=15,b=15),paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig,width='stretch')
                    with r:
                        st.markdown('#### Análisis por fecha')
                        daily=filtered.assign(Fecha=filtered.timestamp.dt.date).groupby('Fecha').size().rename('Análisis').reset_index()
                        fig=px.area(daily,x='Fecha',y='Análisis',color_discrete_sequence=['#57E6C0'])
                        fig.update_layout(height=340,margin=dict(l=0,r=10,t=15,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',yaxis_dtick=1)
                        st.plotly_chart(fig,width='stretch')
                    st.markdown('#### Conteo por categoría')
                    st.dataframe(counts,width='stretch',hide_index=True)
                st.markdown('#### Registros de la selección')
                visible=visible_frame(filtered)
                st.dataframe(visible,width='stretch',hide_index=True,column_config={'Confianza (%)':st.column_config.NumberColumn(format='%.1f %%')})
                st.download_button('Descargar selección CSV',visible.to_csv(index=False).encode('utf-8-sig'),'ecovision_historial_atlas.csv' if USE_ATLAS else 'ecovision_resultados_simulados.csv','text/csv')
                if USE_ATLAS:
                    st.markdown('#### Fotografías asociadas a las clasificaciones')
                    image_rows = filtered.loc[filtered['has_image']].copy()
                    if image_rows.empty:
                        st.info('Los registros filtrados no tienen fotografías guardadas. '
                                'En Clasificador, activa «Guardar fotografía en MongoDB Atlas» '
                                'antes de analizar una nueva imagen.')
                    else:
                        label_by_id = {
                            row['id']: (
                                f"{row['timestamp']:%d/%m/%Y %H:%M} · "
                                f"{row['category']} · {row['confidence']:.1f}% "
                                f"· #{row['id'][:8]}"
                            )
                            for _, row in image_rows.iterrows()
                        }
                        selected_id = st.selectbox(
                            'Selecciona un análisis para visualizar su fotografía',
                            options=list(label_by_id),
                            format_func=lambda record_id: label_by_id[record_id],
                        )
                        if selected_id:
                            selected = image_rows.loc[image_rows['id'] == selected_id].iloc[0]
                            try:
                                img_data = get_prediction_image(selected_id)
                                if img_data is None:
                                    st.warning('La imagen de esta clasificación no está disponible en Atlas.')
                                else:
                                    preview, details = st.columns([1.2, 1], gap='large')
                                    with preview:
                                        st.image(img_data, caption='Fotografía guardada en Atlas', width='stretch')
                                    with details:
                                        st.markdown(f"**Categoría:** {selected['category']}")
                                        st.markdown(f"**Confianza:** {selected['confidence']:.1f}%")
                                        st.markdown(f"**Fecha (Perú):** {selected['timestamp']:%d/%m/%Y %H:%M}")
                                        st.caption(f"Registro: {selected_id}")
                                        st.download_button(
                                            'Descargar fotografía',
                                            data=img_data,
                                            file_name=f'ecovision_{selected_id[:8]}.jpg',
                                            mime='image/jpeg',
                                        )
                            except Exception as exc:
                                st.error('No fue posible recuperar la fotografía desde GridFS.')
                                st.caption('Diagnóstico: ' + type(exc).__name__)

elif page=='Dataset en Atlas':
    from ecovision.atlas_ui import render_atlas
    render_atlas(ROOT)

elif page=='Laboratorio IA':
    if USE_ATLAS:
        hero('03 / MODELOS · MÉTRICAS · PRUEBAS','Laboratorio IA conectado a Atlas.','Consulta los modelos reales y la distribución train/val/test del dataset. El entrenamiento sigue ejecutándose en Google Colab.')
        try:
            from ecovision.atlas_lab import render_atlas_lab
            render_atlas_lab()
        except Exception as exc:
            st.error('No se pudo consultar el Laboratorio IA en Atlas. Comprueba permisos y conectividad.')
            st.caption('Tipo de error: ' + type(exc).__name__)
        with st.expander('Abrir laboratorio local anterior (opcional)'):
            from ecovision.legacy_lab import render_legacy_lab
            render_legacy_lab(ROOT, DATA, hero)
    else:
        from ecovision.legacy_lab import render_legacy_lab
        render_legacy_lab(ROOT, DATA, hero)

else:
    hero('04 / EVIDENCIA · ALCANCE · CONTINUIDAD','Dos sprints. Evidencia verificable.',
         'Validación funcional de módulos y servicios. Las comprobaciones técnicas no constituyen aceptación formal de historias.')
    if USE_ATLAS:
        st.markdown('#### Comprobaciones técnicas de integración')
        if st.button('Verificar conexión y recursos de Atlas'):
            try:
                from ecovision.atlas_service import get_db
                check_connection()
                db = get_db()
                n_images = db['imagenes'].count_documents({})
                n_models = db['modelos'].count_documents({})
                n_preds = db['clasificaciones'].count_documents({'tipo':'Real'})
                c1,c2,c3 = st.columns(3)
                c1.metric('Imágenes en Atlas', f'{n_images:,}')
                c2.metric('Modelos en Atlas', n_models)
                c3.metric('Predicciones reales', n_preds)
                if n_models and n_images:
                    st.success('Conexión y recursos confirmados. Verifica manualmente la inferencia y las historias.')
                else:
                    st.warning('La conexión funciona, pero hay colecciones sin registros.')
            except Exception as exc:
                st.error('No se pudo completar la verificación de Atlas.')
                st.caption('Tipo de error: ' + type(exc).__name__)
    st.warning('Este panel describe el software entregado. La aceptación de cada historia y las capturas reales deben registrarse después de ejecutar las comprobaciones en tu equipo.')
    rows=[
        ('HU-01',1,3,'Interfaz e instrucciones','Clasificador','Revisar propósito, categorías, exclusiones y límites.'),
        ('HU-02',1,3,'Carga y previsualización','Clasificador','Cargar JPG/PNG y comprobar aviso con archivo inválido.'),
        ('HU-03',1,5,'Captura con cámara','Clasificador','Autorizar cámara, capturar foto y probar alternativa con carga.'),
        ('HU-04',1,3,'Historial de consultas','Historial','Generar una consulta y comprobar persistencia al recargar.'),
        ('HU-05',1,3,'Resultado, análisis y limpieza','Clasificador','Analizar; revisar categoría/porcentaje; limpiar sin borrar historial.'),
        ('HU-06',2,3,'Orientaciones y advertencias','Clasificador','Revisar cinco orientaciones y elevar umbral para ver advertencia.'),
        ('HU-07',2,5,'Dashboard','Dashboard','Cargar lote; verificar 20 registros y conteos consistentes.'),
        ('HU-08',2,3,'Filtros combinados','Dashboard','Filtrar categoría, confianza y fechas; restablecer; probar selección vacía.'),
        ('HU-09',2,5,'Preparación de imágenes','Laboratorio IA','Importar ZIP; descargar manifiesto; verificar separación por objeto.'),
        ('HU-10',2,8,'Entrenamiento y guardado','Laboratorio IA','Entrenar, exportar modelo y ejecutar inferencia independiente.'),
    ]
    table=pd.DataFrame(rows,columns=['Historia','Sprint','SP','Función','Pantalla','Comprobación propuesta'])
    if USE_ATLAS:
        pruebas_reales = {
            'HU-04': 'Clasificar una fotografía y comprobar persistencia de la predicción en Atlas.',
            'HU-07': 'Crear predicciones reales y verificar conteos y gráficos en Dashboard.',
            'HU-09': 'Filtrar el dataset de 7,010 imágenes por categoría y subconjunto.',
            'HU-10': 'Leer modelo V1 desde GridFS, comprobar métricas e inferencia real.',
        }
        for historia, prueba in pruebas_reales.items():
            table.loc[table['Historia'] == historia, 'Comprobación propuesta'] = prueba
    a,b=st.columns(2)
    a.metric('Sprint 1 · HU-01 a HU-05','17 SP')
    b.metric('Sprint 2 · HU-06 a HU-10','24 SP')
    st.dataframe(table,width='stretch',hide_index=True)
    st.download_button('Descargar matriz de comprobación CSV',table.to_csv(index=False).encode('utf-8-sig'),'matriz_validacion_sprints.csv','text/csv')
    with st.expander('Sprint 3 · Trabajo pendiente'):
        st.write('HU-11: validación y ajuste. HU-12: evaluación final de inferencia y tiempo de respuesta. HU-13: comprobar integración de Atlas con Clasificador y Dashboard. HU-14: reportes de corrección. HU-15: evaluación con estudiantes y docentes.')
        st.write('La adaptación de las pantallas a computadora y celular se mantiene como criterio transversal. Verifica la cámara y el diseño con tus dispositivos antes de aceptar las historias.')

st.markdown('<div class="footer">ECOVISION / TECNOLOGÍAS EMERGENTES · El usuario mantiene la decisión final. · Historial en hora de Perú (UTC−5).</div>',unsafe_allow_html=True)
