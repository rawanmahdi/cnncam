from keras.applications.xception import Xception
from cnncam import display_heatmap
import numpy as np
import tensorflow as tf

# load pretrained Xception model
model = Xception()

# load and preprocess image
img_path =  "sample_images/meso_grass.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
img = tf.keras.preprocessing.image.img_to_array(img)
# get prediction
pred_img = np.expand_dims(img, axis=0)
pred = np.argmax(model.predict(pred_img))

display_heatmap(model=model, predicted_class=pred, layer_name='block1_conv2',img=img, alpha=1)