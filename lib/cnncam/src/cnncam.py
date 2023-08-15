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

  grad_cam = GradCAM(model=my_cnn, 
                    class_idx=pred, 
                    layer_name=conv_layer_name
                    )

  heatmap = grad_cam.get_heatmap(img)
"""

import numpy as np
import tensorflow as tf


class GradCAM:
    """class that generate GradCAM explanations as heatmaps

    Attributes:
        model: keras.Model containing ?Conv_2D?  layer 
        class_idx: int representing class label, may be true or 
            predicted label 
        layer_name: String of convolutional layer name to explain 
    """
    def __init__(self, model, class_idx, layer_name=None):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name

        if self.layer_name is None:
            for layer in (self.model.layers):
                if len(layer.output_shape) == 4:
                    self.layer_name = layer.name
                else:
                    raise ValueError("Could not find 4D layer, enter layerName")

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
