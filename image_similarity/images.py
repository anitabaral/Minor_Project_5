import os
import pathlib as Path

import cv2

class LoadImage:
  def __init__(self, images_path):
    self.path = images_path

  def get_images(self):

    total_images = {}
    data_path = os.listdir(self.path)
    for sample1 in data_path:
      image_folder = file_path / sample1
      images_path = os.listdir(image_folder)
      images = []
      for sample in images_path:
        image_path = image_folder / sample
        if os.exists(image_path):
          image = cv2.imread(str(image_path))
          images.append(image)
        else:
          raise ValueError('Error while reading images.')
      total_images[sample1] = images
    
      return total_images