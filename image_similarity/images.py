import os
import pathlib as Path

import cv2

class LoadImage:
  def __init__(self, images_path):
    self.path = images_path

  def get_images(self):

    total_images = []  
    data_path = os.listdir(self.path)
    for sample in data_path:
      image_path = path / sample
      if os.exists(image_path):
        image = cv2.imread(str(image_path))
        total_images.append(image)
      else:
        raise ValueError('Error while reading images')
    
    return total_images