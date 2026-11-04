# Verificación del proyecto

## Comprobaciones ejecutadas

Se ejecutaron siete pruebas automáticas con `python -m unittest discover -s tests -v`: todas finalizaron correctamente.

1. Rechazo de imágenes inválidas y consistencia de las probabilidades simuladas.
2. Persistencia SQLite, lote de veinte registros sin duplicación y filtros combinados.
3. Separación por objeto, entrenamiento exclusivo con `train`, guardado y recarga del baseline para inferencia.
4. Rechazo de ZIP inválido y rutas que intentan salir del conjunto.
5. Rechazo de una misma imagen etiquetada en dos categorías diferentes.
6. Recorrido de interfaz con AppTest: ejemplo, análisis, limpieza, dashboard, filtros vacíos, restablecimiento, navegación, carga del ZIP, preparación y entrenamiento del baseline.
7. Carga desde el widget: archivo corrupto, imagen válida, análisis y restablecimiento del resultado cuando cambia la imagen.

El entrenamiento de prueba utilizó las ilustraciones sintéticas incluidas, no fotografías de residuos reales. La prueba que verifica el aislamiento del entrenamiento elimina los archivos de validación y prueba antes de ajustar el baseline; el entrenamiento y la inferencia guardada continúan funcionando.

También se inició el servidor Streamlit y se comprobó la respuesta HTTP 200 `ok` de su endpoint de salud. Esto verifica el arranque del servidor, no una revisión visual del navegador.

## Entorno de las pruebas

- Python 3.12.
- Streamlit 1.66.1.
- NumPy 2.3.5.
- pandas 2.2.3.
- Pillow 12.3.0.
- Plotly 6.9.0.
- scikit-learn 1.8.0.

Las dependencias se ejecutaron en Linux. Las instrucciones de Windows están incluidas, pero no se ejecutó una instalación de Windows en este entorno.

## Comprobaciones que debes completar

- Cámara física y permisos de tu navegador.
- Revisión visual en computadora y vista móvil; legibilidad y ausencia de superposiciones.
- Entrenamiento con tus fotografías reales y revisión de etiquetas y grupos de objetos.
- Entrenamiento opcional de MobileNetV2 con TensorFlow: el código está incluido, pero no se pudo ejecutar aquí por restricciones de descarga del entorno. No se atribuyen resultados a esa arquitectura.
- Confirmación de las orientaciones con las reglas del punto de recolección de la institución.
- Aceptación del equipo, capturas del tablero y evidencias de participación.

La descarga del navegador para revisión visual tampoco estuvo disponible en este entorno. AppTest comprueba el comportamiento y las excepciones de la interfaz; no sustituye una inspección visual ni la prueba de cámara.

## Alcance del resultado

Las HU-01 a HU-08 se demuestran con resultados simulados explícitos. HU-09 prepara y separa datos por objetos; HU-10 permite entrenar y guardar un baseline real y comprobar su carga. La ruta opcional CNN está implementada y pendiente de ejecución en un entorno con sus dependencias. El ejemplo no acredita exactitud de clasificación de residuos ni impacto ambiental.

Las HU-11 a HU-15 siguen pendientes del Sprint 3. La aplicación no presenta como realizadas esas evaluaciones ni genera opiniones ficticias de usuarios o docentes.
