import cv2

from .models import LoadModel
from .images import LoadImage

class Embedding:
  def __init__(self):

    self.model = LoadModel.get_model()
    self.images = LoadImages.get_images()

  def get_feature_vector(self, image):

    resized_image = cv2.resize(image, (224, 224))
    feature_vector = self.model.predict(resized_image.reshape(1, 224, 224, 3))
 
    return feature_vector

  def get_embeddings(self):

    embeddings = {}
    total_images = self.images
    for flower_name, flower_images in total_images.items():
      feature_list = []
      for flower in flower_images:
        feature = self.get_feature_vector(flower_name, flower, model)
        feature_list.append(feature)
      embeddings[flower_name] = feature_list

    return embeddings