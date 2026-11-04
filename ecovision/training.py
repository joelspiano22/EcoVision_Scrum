from __future__ import annotations
import json
import platform
import importlib.metadata
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .core import CATEGORIES, now_lima

FEATURE_VERSION = "rgb16_hist16_grad8_v1"

def features(image):
    im = np.asarray(image.convert('RGB').resize((64,64)), dtype=np.float32)/255.
    thumb = np.asarray(image.convert('RGB').resize((16,16)),dtype=np.float32).ravel()/255.
    hist = np.concatenate([np.histogram(im[:,:,c],bins=16,range=(0,1),density=False)[0]/4096. for c in range(3)])
    gray = im.mean(axis=2)
    gy,gx = np.gradient(gray)
    edge = np.histogram(np.arctan2(gy,gx),bins=8,range=(-np.pi,np.pi),weights=np.hypot(gx,gy))[0]
    edge = edge/(edge.sum()+1e-8)
    return np.concatenate([thumb,hist,edge]).astype(np.float32)

def train_model(dataset_dir, frame, metadata, output_dir, kind='Ligero', epochs=3, progress=None):
    """HU-10: ajusta exclusivamente con train. No consume validation ni test."""
    start = time.perf_counter()
    train = frame.loc[frame.split == 'train']
    if set(train.category) != set(CATEGORIES):
        raise ValueError("El entrenamiento requiere las cinco categorías.")
    run_id = now_lima().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out = Path(output_dir)/run_id
    out.mkdir(parents=True)
    try:
        y = np.array([CATEGORIES.index(c) for c in train.category])
        if kind == 'Ligero':
            vectors=[]
            for i,relative in enumerate(train.path):
                with Image.open(Path(dataset_dir)/relative) as image:
                    vectors.append(features(image))
                if progress: progress((i+1)/len(train)*.6)
            x=np.array(vectors)
            scaler=StandardScaler()
            scaled=scaler.fit_transform(x)
            clf=LogisticRegression(max_iter=1000,random_state=42)
            clf.fit(scaled,y)
            np.savez_compressed(out/'model.npz',mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
            model_file='model.npz'
            history={"training_accuracy":[float(clf.score(scaled,y))]}
            architecture='Regresión logística + rasgos RGB, histogramas y gradientes (baseline; no CNN)'
            if progress: progress(1.)
        else:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ValueError("Instala requirements-cnn.txt para entrenar MobileNetV2.") from exc
            tf.keras.utils.set_random_seed(42)
            paths=[str(Path(dataset_dir)/p) for p in train.path]
            def read(path,label):
                raw=tf.io.read_file(path)
                image=tf.io.decode_png(raw,channels=3)
                return tf.image.resize(image,(224,224)),label
            ds=tf.data.Dataset.from_tensor_slices((paths,y)).map(read,num_parallel_calls=1)
            ds=ds.shuffle(len(paths),seed=42).batch(8).prefetch(1)
            # Descarga los pesos oficiales la primera vez. No se reemplazan por pesos aleatorios si falla.
            base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
            base.trainable=False
            inputs=tf.keras.Input(shape=(224,224,3))
            x=tf.keras.layers.Rescaling(1./127.5,offset=-1)(inputs)
            x=base(x,training=False)
            x=tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs=tf.keras.layers.Dense(5,activation='softmax')(x)
            model=tf.keras.Model(inputs,outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            class Reporter(tf.keras.callbacks.Callback):
                def on_epoch_end(self,epoch,logs=None):
                    if progress: progress((epoch+1)/epochs)
            fitted=model.fit(ds,epochs=epochs,callbacks=[Reporter()],verbose=0)
            model.save(out/'model.keras')
            model_file='model.keras'
            history={"training_accuracy":[float(v) for v in fitted.history['accuracy']],"training_loss":[float(v) for v in fitted.history['loss']]}
            architecture='MobileNetV2 ImageNet congelada + GlobalAveragePooling2D + Dense(5)'
        meta={"id":run_id,"created":now_lima().isoformat(),"kind":kind,"architecture":architecture,
              "categories":CATEGORIES,"model_file":model_file,"feature_version":FEATURE_VERSION,
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"scikit_learn":importlib.metadata.version("scikit-learn")},
              "parameters":({"max_iter":1000,"C":1.0,"solver":"lbfgs","seed":42,"resize":[64,64],"thumbnail":[16,16]} if kind=="Ligero" else {"epochs":epochs,"batch_size":8,"learning_rate":0.001,"base_frozen":True,"weights":"imagenet","input_shape":[224,224,3],"rescaling":"x/127.5-1","seed":42}),
              "train_images":len(train),"train_objects":train.object_id.nunique(),"epochs":epochs if kind!='Ligero' else None,
              "dataset_dir":str(Path(dataset_dir).resolve()),"dataset_zip_sha256":metadata['zip_sha256'],
              "synthetic":metadata.get('synthetic',False),"history":history,
              "seconds":round(time.perf_counter()-start,2),
              "status":"Entrenado. Validación, evaluación final e integración: pendientes del Sprint 3."}
        (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        return out,meta
    except Exception:
        import shutil
        shutil.rmtree(out)
        raise

def predict_saved(run_dir,image):
    run_dir=Path(run_dir)
    meta=json.loads((run_dir/'metadata.json').read_text(encoding='utf8'))
    if meta['kind']=='Ligero':
        with np.load(run_dir/meta['model_file'],allow_pickle=False) as bundle:
            z=(features(image)-bundle['mean'])/bundle['scale']
            scores=z @ bundle['coef'].T + bundle['intercept']
            probs=np.exp(scores-scores.max())
            probs=probs/probs.sum()
            classes=bundle['classes'].astype(int)
        result=np.zeros(5)
        result[classes]=probs
    else:
        import tensorflow as tf
        model=tf.keras.models.load_model(run_dir/meta['model_file'],compile=False)
        x=np.asarray(image.resize((224,224)),dtype=np.float32)[None,...]
        result=model.predict(x,verbose=0)[0]
    idx=int(np.argmax(result))
    return CATEGORIES[idx],float(result[idx]*100),dict(zip(CATEGORIES,(result*100).tolist()))
