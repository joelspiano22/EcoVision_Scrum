import os
import tempfile
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from generar_ejemplo import make_dataset
from ecovision.dataset import prepare_zip

ROOT=Path(__file__).resolve().parents[1]

def button(app,label):
    return next(b for b in app.button if b.label==label)

class UITests(unittest.TestCase):
    def test_upload_validation_and_selection_change(self):
        import io
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            before=os.environ.get('ECOVISION_DATA_DIR')
            os.environ['ECOVISION_DATA_DIR']=tmp
            try:
                app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
                app.file_uploader[0].set_value(('corrupta.png',b'invalid','image/png')).run()
                self.assertTrue(len(app.error)>0)
                self.assertTrue(button(app,'Analizar imagen →').disabled)
                raw=io.BytesIO();Image.new('RGB',(100,100),'red').save(raw,'PNG')
                app.file_uploader[0].set_value(('foto.png',raw.getvalue(),'image/png')).run()
                button(app,'Analizar imagen →').click().run()
                self.assertIsNotNone(app.session_state['result'])
                raw=io.BytesIO();Image.new('RGB',(100,100),'blue').save(raw,'PNG')
                app.file_uploader[0].set_value(('otra.png',raw.getvalue(),'image/png')).run()
                self.assertIsNone(app.session_state['result'])
                self.assertEqual(len(app.exception),0)
            finally:
                if before is None:os.environ.pop('ECOVISION_DATA_DIR',None)
                else:os.environ['ECOVISION_DATA_DIR']=before

    def test_navigation_demo_filter_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            before=os.environ.get('ECOVISION_DATA_DIR')
            os.environ['ECOVISION_DATA_DIR']=tmp
            try:
                app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(button(app,'Analizar imagen →').disabled)
                button(app,'Probar con imagen de ejemplo').click().run()
                button(app,'Analizar imagen →').click().run()
                self.assertIsNotNone(app.session_state['result'])
                self.assertEqual(len(app.exception),0)
                button(app,'Limpiar').click().run()
                self.assertIsNone(app.session_state['result'])
                app.sidebar.radio[0].set_value('Dashboard').run()
                button(app,'Cargar 20 registros de demostración').click().run()
                self.assertEqual(app.metric[0].value,'21')
                app.multiselect[0].set_value([]).run()
                self.assertTrue(any('No hay resultados' in x.value for x in app.info))
                button(app,'Restablecer filtros').click().run()
                self.assertEqual(app.metric[0].value,'21')
                for route in ['Historial','Laboratorio IA','Validación de sprints']:
                    app.sidebar.radio[0].set_value(route).run()
                    self.assertEqual(len(app.exception),0,route)
                app.sidebar.radio[0].set_value('Laboratorio IA').run()
                app.file_uploader[0].set_value(('dataset.zip',make_dataset(),'application/zip')).run()
                button(app,'Preparar y separar imágenes').click().run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(list((Path(tmp)/'datasets').glob('*/manifest.csv')))
                button(app,'Entrenar y guardar modelo').click().run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(list((Path(tmp)/'models').glob('*/model.npz')))
                self.assertTrue(any('Modelo guardado' in x.value for x in app.success))
            finally:
                if before is None:os.environ.pop('ECOVISION_DATA_DIR',None)
                else:os.environ['ECOVISION_DATA_DIR']=before

if __name__=='__main__':unittest.main()
