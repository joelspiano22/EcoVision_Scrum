import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image
from generar_ejemplo import make_dataset
from ecovision.core import CATEGORIES, Store, demo_prediction, filter_frame, open_image
from ecovision.dataset import prepare_zip
from ecovision.training import train_model, predict_saved

class ProjectTests(unittest.TestCase):
    def test_invalid_and_demo(self):
        with self.assertRaises(ValueError):open_image(b'no image')
        image=Image.new('RGB',(50,50),'green')
        p=demo_prediction(image)
        self.assertEqual(p,demo_prediction(image))
        self.assertAlmostEqual(sum(p['probabilities'].values()),100)

    def test_history_and_combined_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Store(Path(tmp)/'history.db')
            self.assertTrue(db.seed());self.assertFalse(db.seed())
            df=Store(Path(tmp)/'history.db').frame()
            self.assertEqual(len(df),20)
            subset=filter_frame(df,['Papel'],(60,100),(df.timestamp.min().date(),df.timestamp.max().date()))
            self.assertTrue((subset.category=='Papel').all())
            self.assertTrue(subset.confidence.between(60,100).all())
            self.assertTrue(filter_frame(df,[],(0,100)).empty)

    def test_group_split_and_reload_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame,meta=prepare_zip(make_dataset(),root/'dataset')
            self.assertEqual(len(frame),100)
            self.assertEqual(frame.object_id.nunique(),50)
            self.assertEqual(frame.groupby('object_id').split.nunique().max(),1)
            self.assertEqual(set(frame.category),set(CATEGORIES))
            self.assertTrue(meta['synthetic'])
            self.assertEqual(set(frame.split),{'train','validation','test'})
            for p in frame.loc[frame.split!='train','path']:
                (root/'dataset'/p).unlink()
            run,report=train_model(root/'dataset',frame,meta,root/'models')
            self.assertTrue((run/'model.npz').exists())
            self.assertEqual(report['train_images'],60)
            path=frame.loc[frame.split=='train','path'].iloc[0]
            with Image.open(root/'dataset'/path) as image:
                cat,confidence,prob=predict_saved(run,image)
            self.assertIn(cat,CATEGORIES)
            self.assertTrue(0<=confidence<=100)
            self.assertAlmostEqual(sum(prob.values()),100,places=5)
            self.assertEqual(json.loads((run/'metadata.json').read_text())['dataset_zip_sha256'],meta['zip_sha256'])

    def test_invalid_zip_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare_zip(b'invalid',Path(tmp)/'bad')
            raw=io.BytesIO()
            with zipfile.ZipFile(raw,'w') as z:z.writestr('../bad.png',b'dummy')
            with self.assertRaises(ValueError):prepare_zip(raw.getvalue(),Path(tmp)/'escape')
            self.assertFalse((Path(tmp)/'bad.png').exists())

    def test_conflicting_labels(self):
        im=Image.new('RGB',(20,20),'red');raw=io.BytesIO();im.save(raw,'PNG')
        zbytes=io.BytesIO()
        with zipfile.ZipFile(zbytes,'w') as z:
            z.writestr('papel/objeto_1/a.png',raw.getvalue())
            z.writestr('vidrio/objeto_1/a.png',raw.getvalue())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'etiquetas diferentes'):
                prepare_zip(zbytes.getvalue(),Path(tmp)/'bad')

if __name__=='__main__':unittest.main()
