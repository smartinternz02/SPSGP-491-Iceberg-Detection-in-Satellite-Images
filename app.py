import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("iceberg_model.h5")

@app.route('/')
def index():
    return render_template('index.html', methods= ['GET'])

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,"uploads",secure_filename(f.filename))
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (75,75))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        #with graph.as_default():
        preds = model.predict_classes(x)
            
            
        pred = preds[0][0]
        if not pred:
            text ="Beware!! Iceberg ahead."
        else:
            text = "You are safe!! It's a Ship"
    print(text)    
    return text


    
if __name__ == '__main__':
    app.run()
        
        
        
    
    
    