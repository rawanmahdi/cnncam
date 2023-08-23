from lib.cnncam.src.cnncam import check_layer_name
from keras.applications.vgg16 import VGG16
import pytest

def test_check_layer_name() -> None:
    model = VGG16()
    layer_name = 'block3_pool'
    with pytest.raises(TypeError):
        check_layer_name(model, layer_name, base_model=None)
