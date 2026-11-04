from pathlib import Path

import tomli
from pymongo import MongoClient
from pymongo.errors import PyMongoError

ruta = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"

with ruta.open("rb") as archivo:
    configuracion = tomli.load(archivo)

mongodb = configuracion["mongodb"]

try:
    with MongoClient(
        mongodb["uri"],
        serverSelectionTimeoutMS=10000,
    ) as cliente:
        cliente.admin.command("ping")
        print("Conexión con MongoDB Atlas exitosa.")

        base_datos = cliente[mongodb["database"]]
        # Comprueba que puedes consultar la base del proyecto.
        base_datos["imagenes"].find_one({}, {"_id": 1})
        print("Acceso de lectura a la base ecovision confirmado.")

except PyMongoError as error:
    print("No se pudo completar la prueba.")
    print("Tipo de error:", type(error).__name__)
    print("Revisa las credenciales, los permisos y el acceso de red.")