import os


def get_input_images(folder_path):
    """Store all the images in a list, present in the folder path.

    Args:
        folder_path (str): Path of the folder where all the images reside.

    Returns:
        object: List of all the images present in the folder_path.
    """
    list_of_all_input_images = []
    for filename in os.listdir(folder_path):
        list_of_all_input_images.append(filename)

    return list_of_all_input_images
