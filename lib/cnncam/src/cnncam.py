"""A module to generate GradCAM explanations for CNNs.

Gradient-based Class Activation Maps, proposed by Selvaraju et al.
(paper below), computes attribution maps through the gradient of
the predicted class flowing through a specific convolutional layer in
a CNN. Our implementation currently only supports keras models. 

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., 
&amp; Batra, D. (2017). Grad-cam: Visual explanations from deep
Networks via gradient-based localization. 2017 IEEE International
Conference on Computer Vision (ICCV). 
https://doi.org/10.1109/iccv.2017.74 

Typical usage example:

  to get a matrix of numbers representing your heatmap:
  
  from cnncam import GradCAM 

  grad_cam = GradCAM(model=my_cnn, 
                    class_idx=pred, 
                    layer_name=conv_layer_name
                    )

  heatmap = grad_cam.get_heatmap(img)
   
  or to quickly and easily display a heatmap:
  
  from cnncam import display_heatmap
  
  display_heatmap(model=my_cnn,
                  predicted_class=pred, 
                  layer_name=layer_name,
                  img)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

class GradCAM:
    """class that generate GradCAM explanations as heatmaps

    Attributes:
        model: keras.model.Model containing at least one keras.layers.Conv2D layer 
        class_idx: int representing predicted class label 
        layer_name: str of convolutional layer name to explain 
        base_model: str of base model if your conv layer is within a base model, defaults to None 
    """
    def __init__(self, model, class_idx, layer_name, base_model=None):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        self.base_model = base_model
        # TODO: add to docs that base model layers must be inputted and differently
        check_layer_name(model=self.model,
                        layer_name=self.layer_name,
                        base_model=self.base_model)

    def get_heatmap(self, img, eps=1e-8):
        """_summary_

        Args:
            img (numpy.ndarray): array of size (224,224,224,3) 
                representing image to explain output of
            eps (float, optional): ????. Defaults to 1e-8.

        Returns:
            numpy.ndarray: heatmap
        """
        # Add dim to img
        img = np.expand_dims(img, axis=0)

        # Build model that maps image to activations of conv layer of model
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs, # img inputs
            outputs=[self.model.get_layer(self.layer_name).output,
                    self.model.output
            ]
        )

        # Compute gradient of top predicted class for img with respect to
        # Activation of last convolutional layer
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img)
            loss = preds[:, self.class_idx]
        grads = tape.gradient(loss, conv_output) # Gradient of output class

        # Vector of mean intensity of gradient over feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        # Get heatmap by summing all the channels after multiplying
        # feature map by feature importance for top predicted class
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Normalize heatmap
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

# TODO: def function to get last conv layer such that user can enter layer_name='last' instead of a proper layer name


def check_layer_name(model, layer_name, base_model):

    if base_model == None:
        layer_names = [layer.name for layer in model.layers]
        if layer_name in layer_names:
            if not isinstance(model.get_layer(layer_name),
                                keras.layers.convolutional.conv2d.Conv2D):
                raise TypeError('The layer name you entered is not of type tensorflow.keras.layers.Conv2D. Please enter a Conv2D layer name.')
        else:
            raise ValueError('The layer name you entered could not be found in you model, please enter a valid Conv2D layer name.')
    else: 
        layer_names = [layer.name for layer in model.get_layer(base_model).layers] 
        if layer_name in layer_names:
            if not isinstance(model.get_layer(base_model).get_layer(layer_name),
                                keras.layers.convolutional.conv2d.Conv2D):
                raise TypeError('The layer name you entered is not of type tensorflow.keras.layers.Conv2D. Please enter a Conv2D layer name.')
        else:
            raise ValueError('The layer name you entered could not be found in your base model, please enter a valid Conv2D layer name.')
  

def display_heatmap(model, img, predicted_class, layer_name, alpha=0.6, eps=1e-8):
    """function to display heatmap overlayed onto image

    Args:
        model (keras.models.Model): your model containing a convolutional layer
        TODO: define required size of input image
        img (ndarray): _description_
        predicted_class (int): your models prediction for the input image
        layer_name (str): name of a keras.layers.Conv2D layer within your model to be explained
        eps (_type_, optional): _description_. Defaults to 1e-8.
    """
    # Get GradCAM heatmap
    gradcam = GradCAM(model=model, class_idx=predicted_class, layer_name=layer_name)
    heatmap = gradcam.get_heatmap(img, eps=eps)

    # Configure plot
    heatmap = cv2.resize(heatmap, (224,224))
    extent = 0,224,0,224
    fig = plt.figure(frameon=False) 
    # TODO : do i need the above line?
    plt.imshow(np.squeeze(img).astype(np.uint8), extent=extent)
    plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=alpha, extent=extent)
    # Display plot
    plt.show()
