import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from cnncam import GradCAM, display_heatmap

# Load pretrained VGG-16 model
vgg16 = VGG16(include_top=True, weights="imagenet")
vgg16.summary()

# Load and preprocess sample image
img = tf.keras.preprocessing.image.load_img("sample_images/meso_grass.jpg", target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Get model's prediction
pred = np.argmax(vgg16.predict(img))

# Instantiate GradCAM with parameters and get grid values of heatmap
gradcam = GradCAM(model=vgg16, class_idx=pred, layer_name='block5_conv2')
heatmap = gradcam.get_heatmap(img=tf.squeeze(img))

# Or display heatmap directly
display_heatmap(model=vgg16, predicted_class=pred, layer_name='block5_conv3', img=tf.squeeze(img))
