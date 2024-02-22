import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
import numpy as np
from tensorflow.keras.preprocessing import image
model=tf.keras.models.load_model('model.h5')

# Assuming your file is in the same directory as the Jupyter Notebook
file_name = 'image_name.png'  # Replace with the actual file name

# Construct the path to the file
path = os.path.join(os.getcwd(), file_name)

# Predicting images
img = image.load_img(path, target_size=(300, 300)) #dimensions match huna parcha
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
predictions=np.argmax(classes)
labels=['Covid','Normal','Viral Pneumonia']
print(labels[predictions])