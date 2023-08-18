#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from cnncam import GradCAM

# Load pretrained VGG-16 model
model = VGG16(include_top=True, weights="imagenet")
model.summary()
#%%
# Load and preprocess sample image
img = tf.keras.preprocessing.image.load_img("sample_images/meso_table.jpg", target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Get model's prediction
pred = np.argmax(model.predict(img))

# Instantiate GradCAM with parameters
gradcam = GradCAM(model=model, classIdx=pred, layerName='block5_conv3')
heatmap = gradcam.get_gradcam_heatmap(img=img)

