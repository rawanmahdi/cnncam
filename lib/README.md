# cnncam
<!-- badges: start -->
![Maintainer](https://img.shields.io/badge/maintainer-rawanmahdi-pink)
[![PyPI version](https://badge.fury.io/py/cnncam.svg)](https://badge.fury.io/py/cnncam)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint) [![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit/) 
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![GitHub release](https://img.shields.io/github/release/rawanmahdi/cnncam.svg)](https://GitHub.com/rawanmahdi/cnncam/releases/)
<!-- [![DOI](https://joss.theoj.org/papers/10.21105/joss.02027/status.svg)](https://doi.org/10.21105/joss.02027)
 -->
<!-- badges: end -->

## Introduction

An open python library for researchers and developers to generate GradCAM explanations for the tensorflow CNN models. 

Other popular python ML frameworks will soon be supported by the cnn-cam library

### Install

The below instructions assume you already have `pip` installed and exposed to the python environment where you want to run `cnncam`. 
Official instructions for installing `pip` can be found [here](https://pip.pypa.io/en/stable/installation/)

Run the below pip command in a shell of your choice. 
```
pip install cnncam
```

### Demo

We currently support two ways of obtaining GradCAM heatmaps:

1. Display heatmap.
If you're only interested in seeing the heatmap images for your model's prediction, you can run the below script:

```python
from cnncam import display_heatmap
from keras.applications.vgg16 import VGG16

display_heatmap(model=VGG16(), # your keras model
                img=img, # your image
                predicted_class=pred, # your models prediction for the image 
                layer_name='block5_conv3', # the layer you would like to see GradCAM for 
                alpha=0.6 # opacity of heatmap overlayed on image
                )
```
Output: 
![alt text](https://github.com/rawanmahdi/cnncam/blob/main/examples/output_images/meso_grass_vgg16.png?raw=true)

We can observe the differences in convolutional layer behaviours across different models, for example, the below code executes the same function on Xception instead of VGG16:

```python
from cnncam import display_heatmap
from keras.applications.xception import Xception

display_heatmap(model=Xception(), # your keras model
                img=img, # your image
                predicted_class=pred, # your models prediction for the image 
                layer_name='block5_conv3', # the layer you would like to see GradCAM for 
                alpha=1 # opacity of heatmap overlayed on image
                )
```
Output:
![alt text](https://github.com/rawanmahdi/cnncam/blob/main/examples/output_images/meso_grass_xception.png?raw=true)

See `/examples` for executable examples, including the above application of our implementation of GradCAM on VGG-16 with the owner of this repo's very cute cat, meso.
