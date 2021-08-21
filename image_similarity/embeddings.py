
import cv2

from .load_model import LoadModel
from .load_images import LoadImages

class Embeddings:
  def __init__(self):

    self.model = LoadModel.get_model()
    self.images = LoadImages.get_images()

  def get_feature_vector(self, image):

  resized_image = cv2.resize(image, (224, 224))
  feature_vector = self.model.predict(resized_image.reshape(1, 224, 224, 3))
 
  return feature_vector

  def get_embeddings(self):

    embeddings = []
    total_images = self.images
    for index, image in enumerate(total_images):
      feature = self.get_feature_vector(image)
      embeddings.append(feature)

    return embeddings