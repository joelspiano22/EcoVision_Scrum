"""Laboratorio local anterior: conservado para pruebas y retrocompatibilidad."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from .core import CATEGORIES, open_image
from .dataset import prepare_zip, load_dataset
from .training import train_model, predict_saved

def render_legacy_lab(ROOT, DATA, hero):
    hero('03 / PREPARA · ENTRENA · GUARDA','Del conjunto de imágenes al modelo.','Espacio técnico del Sprint 2. Organiza fotografías por objeto y entrena un modelo guardado en disco. La interfaz pública continúa en modo demostración.')
    st.info('El baseline ligero permite probar el flujo sin TensorFlow. Para la CNN del proyecto, selecciona MobileNetV2 e instala requirements-cnn.txt. Validación, prueba final e integración se reservan para el Sprint 3.')
    tab1,tab2,tab3=st.tabs(['01 · Preparar imágenes','02 · Entrenar modelo','03 · Comprobar carga'])
    with tab1:
        st.markdown('### Un objeto, todas sus vistas, una carpeta')
        st.write('Carga un ZIP con las cinco categorías. Cada objeto físico debe tener su propia carpeta; todas sus fotografías deben estar juntas. Mínimo técnico: cinco objetos distintos por categoría. Este mínimo permite ejecutar el flujo, no demuestra suficiencia estadística.')
        st.code('papel/objeto_01/frente.jpg\npapel/objeto_01/reverso.jpg\npapel/objeto_02/frente.jpg\ncarton/objeto_01/frente.jpg\nplastico/objeto_01/frente.jpg\nvidrio/objeto_01/frente.jpg\nmetal/objeto_01/frente.jpg',language='text')
        demo_zip=ROOT/'ejemplos/dataset_sintetico_demo.zip'
        if demo_zip.exists():
            st.download_button('Descargar ZIP sintético para ensayar',demo_zip.read_bytes(),demo_zip.name,'application/zip')
        st.caption('El ejemplo contiene ilustraciones artificiales. No son fotografías de residuos ni sirven para sustentar desempeño en la institución.')
        upload=st.file_uploader('Conjunto de imágenes ZIP · Hasta 60 MB',type=['zip'],key='dataset_zip')
        seed=st.number_input('Semilla de separación',min_value=0,max_value=999999,value=42,step=1)
        if st.button('Preparar y separar imágenes',type='primary',disabled=upload is None):
            try:
                raw=upload.getvalue()
                target=DATA/'datasets'/(hashlib.sha256(raw).hexdigest()[:16]+f'_{seed}')
                with st.spinner('Revisando imágenes y agrupando objetos…'):
                    frame,meta=prepare_zip(raw,target,int(seed))
                st.session_state.active_dataset=str(target)
                st.success('Conjunto preparado. Los grupos de objetos no se comparten entre entrenamiento, validación y prueba.')
            except (ValueError,OSError,RuntimeError) as exc:
                st.error(str(exc))
        datasets=sorted((DATA/'datasets').glob('*/manifest.csv')) if (DATA/'datasets').exists() else []
        if datasets:
            paths=[str(p.parent) for p in datasets]
            active=st.session_state.get('active_dataset',paths[-1])
            selected=st.selectbox('Conjunto preparado',paths,index=paths.index(active) if active in paths else len(paths)-1,format_func=lambda x:Path(x).name)
            st.session_state.active_dataset=selected
            frame,meta=load_dataset(selected)
            if meta['synthetic']:st.warning('CONJUNTO SINTÉTICO · Solo para pruebas del software.')
            c1,c2,c3=st.columns(3)
            c1.metric('Imágenes válidas',meta['images']);c2.metric('Objetos declarados',meta['objects']);c3.metric('Duplicados retirados',meta['duplicates_removed'])
            counts=pd.crosstab(frame.category,frame.split).reindex(CATEGORIES).rename_axis('Categoría')
            st.dataframe(counts,width='stretch')
            st.caption('Separación aproximada 60/20/20 por objetos, dentro de cada categoría. Las cantidades de imágenes pueden variar según las vistas de cada objeto.')
            st.warning(meta['limitation'])
            st.download_button('Descargar manifiesto CSV',frame.to_csv(index=False).encode('utf-8-sig'),'manifest.csv','text/csv')
            st.download_button('Descargar configuración del conjunto',json.dumps(meta,ensure_ascii=False,indent=2),'dataset_metadata.json','application/json')
    with tab2:
        dataset=st.session_state.get('active_dataset')
        kind=st.selectbox('Arquitectura',['Ligero','MobileNetV2'],format_func=lambda x:'Baseline ligero · Regresión logística' if x=='Ligero' else 'CNN · MobileNetV2 con transferencia de aprendizaje')
        epochs=st.slider('Épocas de MobileNetV2',1,20,3,disabled=kind=='Ligero')
        st.caption('El baseline usa características RGB, histogramas y gradientes; no es una CNN. MobileNetV2 utiliza pesos ImageNet y entrena una capa de salida para las cinco categorías. Requiere conexión la primera vez para obtener los pesos.')
        if not dataset:st.info('Primero importa y prepara el ZIP en la pestaña anterior.')
        else:
            frame,meta=load_dataset(dataset)
            if meta['synthetic']:st.warning('Estás por entrenar con ilustraciones sintéticas. El resultado solo demuestra la ejecución del código.')
            st.write(f'Entrenamiento: **{int((frame.split=="train").sum())} imágenes**. Validación y prueba permanecen reservadas.')
        if st.button('Entrenar y guardar modelo',type='primary',disabled=not dataset):
            bar=st.progress(0.,text='Entrenamiento en curso')
            try:
                with st.spinner('Entrenando; espera hasta que aparezca la confirmación…'):
                    run,report=train_model(dataset,frame,meta,DATA/'models',kind,epochs,lambda p:bar.progress(p))
                st.session_state.last_run=str(run)
                st.success(f'Modelo guardado: {run.name}. Puedes comprobar su carga en la siguiente pestaña.')
            except Exception as exc:
                st.error(f'No se completó el entrenamiento: {exc}')
                st.caption('Para MobileNetV2, comprueba la instalación de TensorFlow y la descarga de pesos oficiales. El baseline funciona sin TensorFlow.')
    with tab3:
        runs=sorted((DATA/'models').glob('*/metadata.json'),reverse=True) if (DATA/'models').exists() else []
        if not runs:st.info('Aún no hay modelos guardados. Ejecuta un entrenamiento en la pestaña anterior.')
        else:
            selected=st.selectbox('Modelo guardado',[str(p.parent) for p in runs],format_func=lambda p:Path(p).name)
            report=json.loads((Path(selected)/'metadata.json').read_text(encoding='utf8'))
            if report['synthetic']:st.warning('Modelo entrenado con datos sintéticos: no representa desempeño en residuos reales.')
            st.write(f'**Arquitectura:** {report["architecture"]}')
            st.write(f'**Imágenes de entrenamiento:** {report["train_images"]} · **Tiempo:** {report["seconds"]} s')
            history=report['history']
            curve=pd.DataFrame({'Paso':range(1,len(history['training_accuracy'])+1),'Exactitud en entrenamiento':history['training_accuracy']})
            st.dataframe(curve,width='stretch',hide_index=True)
            if len(curve)>1:st.line_chart(curve.set_index('Paso'))
            st.caption('Este indicador se calcula sobre los datos usados para ajustar el modelo. No es exactitud de validación ni prueba, y no mide generalización.')
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in Path(selected).iterdir():z.write(p,p.name)
            st.download_button('Exportar modelo y metadatos ZIP',archive.getvalue(),f'modelo_{Path(selected).name}.zip','application/zip')
            st.download_button('Descargar registro del entrenamiento',json.dumps(report,ensure_ascii=False,indent=2),'training_metadata.json','application/json')
            test_image=st.file_uploader('Comprobar carga con una fotografía JPG o PNG',type=['jpg','jpeg','png'],key='lab_predict')
            if st.button('Ejecutar predicción independiente',disabled=test_image is None):
                try:
                    image=open_image(test_image.getvalue())
                    category,confidence,probabilities=predict_saved(selected,image)
                    st.image(image,width=260)
                    st.success(f'Predicción del modelo entrenado: {category} · {confidence:.1f}%')
                    st.dataframe(pd.DataFrame({'Categoría':probabilities.keys(),'Probabilidad (%)':probabilities.values()}),hide_index=True)
                    st.caption('Comprobación individual de carga e inferencia. No constituye evaluación técnica ni se guarda en el historial de demostración.')
                except Exception as exc:st.error(f'No se pudo ejecutar la predicción: {exc}')

