from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

class LoadModel:
  def __init__(self):
    pass

  def get_model(self):

    base_model = VGG16(input_shape = (224, 224, 3), include_top = True, weights = 'imagenet', pooling = 'max')
    model = Model(inputs = base_model.input, outputs = base_model.get_layer('fc2').output)

    return model