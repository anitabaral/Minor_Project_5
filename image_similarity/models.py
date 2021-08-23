import yaml
from keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


def get_vgg16_model():
    base_model = VGG16(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet",
        pooling="max",
    )
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)
    return model, base_model


def get_vgg16_flower_model():
    vgg16_flower_model_path = file_paths["vgg16_flower_model_path"]
    base_model = load_model(vgg16_flower_model_path)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense").output)
    return model, base_model
