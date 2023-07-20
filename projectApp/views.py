from django.shortcuts import render 
from tensorflow import keras

from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

from PIL import Image 
import numpy as np 
import os
from django.core.files.storage import FileSystemStorage
from efficientnet.tfkeras import EfficientNetB5
from keras.utils import custom_object_scope




# Create your views here,
media='media'
#model = load_model('mod-sigmoidfocalentropy false-ef5.h5')

#model = load_model('thisModel.h5')
#with keras.utils.custom_object_scope({'RectifiedAdam': tfa.optimizers.RectifiedAdam}):
    #model = load_model('thisModel.h5')

with keras.utils.custom_object_scope({'RectifiedAdam': tfa.optimizers.RectifiedAdam}):
    model = keras.models.load_model('model_tf')


#with custom_object_scope({'FixedDropout': FixedDropout}):
 #   model = load_model('thisModel.h5')

def makepredictions(path):
    #we open the image
    img=Image.open(path)
    #we resize the image for model
    img_d = img.resize ( (256, 256))
    # we check if image is RGB or not
    if len(np.array(img_d).shape)<4:
        rgb_img =Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d

    # here we convert the image into numpy array and reshape
    rgb_img=np.array(rgb_img, dtype=np.float64) 
    rgb_img=rgb_img.reshape (1, 256, 256, 3)
    #we make predictions here
    predictions =model.predict(rgb_img)
    #predicted_class = 1 if predictions >= 0.5 else 0


    probability_class1 = predictions[0][0]*100
    
    probability_class0 = 100 - probability_class1

    # We return the probabilities for both classes
    return round(probability_class0), round(probability_class1)











def index(request):
    if request.method == "POST" and request.FILES['upload']:
        if 'upload' not in request.FILES:
            return render(request, 'index.html')
        f = request.FILES[ 'upload' ]
        if f =='':
            return render(request, 'index.html')
        upload =request.FILES[ 'upload' ]
        fss = FileSystemStorage()
        file=fss.save(upload.name, upload) 
        file_url=fss.url(file)
        
        predictions0, prediction1 = makepredictions(os.path.join (media, file)) 
        print(predictions0, prediction1)
        return render(request, 'index.html', {'pred0':predictions0,'pred1':prediction1, 'file_url':file_url})
    return render(request, 'index.html')


