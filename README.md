# EcoVision — Streamlit + MongoDB Atlas (V1)

Esta versión conserva el diseño oscuro y las seis secciones de `app.py`, pero conecta el flujo principal a MongoDB Atlas. El dataset (7,010 imágenes), los archivos de GridFS y el modelo MobileNetV2 V1 **no se incluyen en el ZIP** porque ya están en tu clúster `ecovision`. El proyecto anterior y su laboratorio local permanecen disponibles cuando no se configura Atlas; su descripción está en `README_ANTERIOR.md`.

## Instalación recomendada (Windows)

Puedes trabajar en una copia del proyecto actual. **No reemplaces ni publiques tus secretos.** Si ya tienes un entorno `.venv` puedes reutilizarlo; si no, créalo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-cnn.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La instalación de `requirements-cnn.txt` (TensorFlow) es necesaria para **inferencias reales** en Clasificador y Laboratorio IA; no necesitas volver a entrenar el modelo en tu computadora. También puedes ejecutar `iniciar_windows.bat` una vez instalado el entorno.

## Configurar MongoDB Atlas

Copia `.streamlit/secrets.example.toml` a `.streamlit/secrets.toml` y reemplaza `USUARIO`, `CLAVE` y `HOST` con los datos privados de tu clúster. Si ya tienes un `secrets.toml` válido en tu proyecto original, **conserva ese archivo** y no crees otro. No pongas las credenciales dentro de `app.py` y no subas `secrets.toml` a GitHub. Alternativamente, puedes usar variables de entorno `MONGO_URI` y `MONGO_DB=ecovision`.

```toml
[mongodb]
uri = "mongodb+srv://USUARIO:CLAVE@HOST.mongodb.net/?retryWrites=true&w=majority"
database = "ecovision"
```

### Colecciones utilizadas

| Colección | Función |
| --- | --- |
| `imagenes` | Metadatos y división `train` / `val` / `test`. |
| `fs.files`, `fs.chunks` | Imágenes y archivo `.keras` en GridFS. |
| `modelos` | Datos del modelo V1, `gridfs_id`, `sha256`, orden de clases y métricas. |
| `clasificaciones` | Se crea al guardar la primera predicción real. |
| `importaciones_control` | Se conserva sin cambios. |

Requisitos del documento `modelos`: `version`, `gridfs_id`, `categorias` (en el orden exacto del entrenamiento), `img_size=[224,224]`, `preprocesamiento="mobilenet_v2.preprocess_input"`, `sha256`, `metricas`. Estos campos corresponden al registro creado en Colab en el paso 16.

**Importante:** `st.secrets` en tu computadora es independiente de los Secrets de Google Colab. Aquí debes proporcionar una cuenta con permisos mínimos necesarios: lectura de `imagenes`, `fs.files`, `fs.chunks`, `modelos`, y escritura de `clasificaciones`.

## Uso de los seis módulos

1. **Clasificador:** recupera el modelo V1 desde GridFS una vez por proceso, lo carga en memoria (copia temporal eliminada al terminar la carga), realiza predicciones reales y registra `categoria`, `confianza`, `probabilidades`, versión, fecha, origen y hash en `clasificaciones`. No guarda la fotografía subida.
2. **Dashboard:** usa únicamente los registros reales en `clasificaciones`, con filtros y visualización por fecha.
3. **Historial:** lee `clasificaciones` y permite exportar un CSV. Se muestran los 5,000 registros más recientes; las fechas se convierten a la zona horaria de Lima.
4. **Dataset en Atlas:** lee `imagenes`, muestra distribución y filtra por categoría y subconjunto. Recupera las miniaturas desde GridFS solo cuando activas la vista previa.
5. **Laboratorio IA:** consulta los registros en `modelos` y el número de imágenes por `train`, `val` y `test`. Permite probar inferencia sin crear un historial. El antiguo laboratorio de ZIP y modelos locales permanece dentro de un desplegable.
6. **Validación de sprints:** conserva la tabla original y agrega comprobaciones de conexión y conteos en Atlas. Las verificaciones no equivalen a la aprobación docente de historias.

Si no se configura Atlas, el programa conserva el comportamiento de demostración original, incluyendo SQLite y el laboratorio local. Si la conexión falla **no pasa silenciosamente a predicciones simuladas**: muestra un error seguro.

## Orden recomendado para probar en tu equipo

1. En **Dataset en Atlas**, confirma que aparecen 7,010 imágenes y los tres subconjuntos.
2. En **Laboratorio IA**, comprueba que la versión `v1` tenga métricas de prueba y el orden de categorías `Cartón, Metal, Papel, Plástico, Vidrio`.
3. En **Clasificador**, sube una fotografía válida de un solo residuo y pulsa *Analizar imagen*. La primera predicción puede tardar porque carga el archivo `.keras` en memoria.
4. En **Historial** y **Dashboard**, comprueba que la clasificación y sus cifras aparecen.
5. En **Validación de sprints**, pulsa *Verificar conexión y recursos de Atlas*.

## Seguridad y notas técnicas

- MongoDB almacena el modelo y las imágenes, pero la **inferencia se ejecuta en Streamlit**. La primera vez que se carga el modelo, sus bytes se transfieren desde Atlas; no es posible ejecutar TensorFlow sin disponer de los pesos en memoria.
- La exactitud del 86.69 % fue obtenida en el conjunto de prueba del notebook. No significa que cada predicción tenga 86.69 % de probabilidad de ser correcta.
- Las curvas de entrenamiento y la matriz de confusión no están en los metadatos de `modelos` guardados en el paso 16; no se muestran ni se inventan. Para presentarlas automáticamente hay que almacenar esos resultados en Atlas después.
- El ZIP excluye deliberadamente `.venv`, el dataset original, archivos SQLite y `.streamlit/secrets.toml`; no se perderán los archivos de tu proyecto anterior si los conservas.
- Si compartiste una URI con credenciales válidas en un ZIP o repositorio, considera rotar la contraseña del usuario de Atlas.

## Verificación disponible

La estructura Python fue revisada con `compileall` y pruebas estáticas. No se validó el funcionamiento conectado en tu Atlas ni el arranque de Streamlit en este entorno: debes ejecutar el recorrido indicado arriba en tu propia computadora.
