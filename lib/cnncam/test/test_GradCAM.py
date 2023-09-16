from lib.cnncam.src.cnncam import GradCAM
from keras.applications.vgg16 import VGG16


def test_gradcam_attributes() -> None:
    gc = GradCAM()