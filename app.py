from image_similarity import Embedding, FeaturesCsv, LoadImage, LoadModel

import yaml

with open("config.yaml", 'r') as stream:
  file_paths = yaml.safe_load(stream)

def main():
   LoadImage(file_paths['images_loc'])
   FeaturesCsv(file_paths['csv_loc'])