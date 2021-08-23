import yaml
from keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


def get_vgg16_model():
    """Loading the pretrained vgg16 model trained on imagenet and extracting the fc2 layer.

    Returns:
        base_model (object): Instance of vgg16 model already trained on imagenet.
        model (object): Instance of vgg16 that outputs the features of fully connected layer 2, taking the input image.
    """
    base_model = VGG16(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet",
        pooling="max",
    )
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

    return model, base_model


def get_vgg16_flower_model():
    """Loading the pretrained vgg16 model trained on flower dataset and extracting the dense layer.

    Returns:
        base_model (object): Instance of vgg16 model already trained on flower dataset.
        model (object): Instance of vgg16 that outputs the features of dense layer, taking the input image.
    """
    vgg16_flower_model_path = file_paths["vgg16_flower_model_path"]
    base_model = load_model(vgg16_flower_model_path)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("dense").output)

    return model, base_model
