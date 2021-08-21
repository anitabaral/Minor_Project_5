import os
import pathlib as Path

import cv2

class LoadImage:
  def __init__(self):
    pass

  def get_images(self):

    total_images = []
    file_path = Path('/content/drive/MyDrive/Leapfrog_internship/Project 5/Project_5')
    data_path = os.listdir(file_path)
    for sample in data_path:
      image_path = file_path / sample
      try: 
        image = cv2.imread(str(image_path))
        total_images.append(image)
      except:
        print('It is not an image')
    
    return total_images