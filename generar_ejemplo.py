"""Crea ilustraciones artificiales para probar importación y entrenamiento."""
import io
import random
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent

def make_dataset():
    out=io.BytesIO()
    categories=['papel','carton','plastico','vidrio','metal']
    colors=['#e5e9e1','#bc8853','#50cda6','#90b9e7','#a6b1bc']
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('DATOS_SINTETICOS.txt','Ilustraciones artificiales, no residuos reales. Únicamente para ensayar el software. No utilizar sus métricas como resultados del proyecto.')
        for c,(cat,color) in enumerate(zip(categories,colors)):
            for obj in range(10):
                for view in range(2):
                    rng=random.Random(c*1000+obj*10+view)
                    im=Image.new('RGB',(128,128),tuple(rng.randint(20,65) for _ in range(3)))
                    d=ImageDraw.Draw(im)
                    x,y=rng.randint(18,32),rng.randint(14,26)
                    if c==0:d.polygon([(x,y),(100,y+4),(96,110),(x+4,106)],fill=color)
                    elif c==1:
                        d.rectangle((x,y+20,105,108),fill=color)
                        d.line((x,58,105,58),fill='#5d3a23',width=4)
                    elif c in [2,3]:
                        d.rounded_rectangle((x+15,y+15,91,112),radius=13,fill=color)
                        d.rectangle((x+28,y,79,y+20),fill=color)
                    else:
                        d.rounded_rectangle((x+7,y+10,98,110),radius=12,fill=color)
                        d.ellipse((x+8,y+10,97,y+30),fill='#dbe1e5')
                    d.line((5,obj*7+5,13,obj*7+7),fill=(rng.randint(80,255),90,180),width=2)
                    raw=io.BytesIO();im.save(raw,'PNG')
                    z.writestr(f'{cat}/objeto_{obj+1:02d}/vista_{view+1}.png',raw.getvalue())
    return out.getvalue()

if __name__=='__main__':
    target=ROOT/'ejemplos/dataset_sintetico_demo.zip'
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(make_dataset())
    print(target)
