import yaml
import logging

from pca import PrincipleComponent
from image_similarity import Embedding, cosine_similar, euclidean_dissimilar

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


if __name__ == "__main__":
    """Generates embeddings for all the images present in the given file path."""
    df_embeddings = Embedding(
        file_paths["input_images_folder_path"], file_paths
    ).get_embeddings()
    logging.info(df_embeddings.head())

    similairity_score = df_embeddings_copy = df_embeddings.copy()
    dissimilarity_score = df_embeddings.drop(labels=0, axis=1, inplace=True)

    """Prints out the similarity and dissimilarity score between any given feature vectors"""
    cosine_similar(df_embeddings)
    euclidean_dissimilar(df_embeddings)

    """Latent Representation of embeddings of the images."""
    PrincipleComponent(df_embeddings, df_embeddings_copy).visualize_pca()
