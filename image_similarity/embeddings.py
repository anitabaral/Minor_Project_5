import yaml
import numpy as np
import pandas as pd
from .images import get_input_images
from .models import get_vgg16_flower_model, get_vgg16_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Embedding:
    def __init__(self, input_images_folder_path, file_paths):
        self.model, self.base_model = get_vgg16_flower_model()
        self.images_name_list = get_input_images(input_images_folder_path)
        self.input_images_folder_path = input_images_folder_path
        self.class_labels = file_paths["class_labels"]

    @staticmethod
    def get_flattern_list(feature_vector_list):
        flat_feature_vector_list = []
        for sublist in feature_vector_list:
            for item in sublist:
                flat_feature_vector_list.append(item)
        return flat_feature_vector_list

    def get_feature_vector(self, image):
        image = load_img(image, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image_array.reshape(1, 224, 224, 3)
        feature_vector = self.model.predict(image_array)
        result = np.argmax(self.base_model.predict(image_array))
        feature_label = [key for key in self.class_labels][result]

        return feature_vector, feature_label

    def get_embeddings(self):

        total_images = self.images_name_list
        feature_list = []
        for image in total_images:
            feature_vector, feature_label = self.get_feature_vector(
                self.input_images_folder_path / image
            )
            feature_vector_list = feature_vector.tolist()
            feature_vector_list = self.get_flattern_list(feature_vector_list)
            feature_vector_list.insert(0, feature_label)
            feature_list.append(feature_vector_list)
        columns_name = np.arange(0, 513)
        df_embeddings = pd.DataFrame(feature_list, columns=columns_name)
        return df_embeddings
