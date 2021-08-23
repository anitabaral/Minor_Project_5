import os


def get_input_images(folder_path):
    list_of_all_input_images = []
    for filename in os.listdir(folder_path):
        list_of_all_input_images.append(filename)
    return list_of_all_input_images
