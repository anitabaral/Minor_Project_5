import yaml

from pca import PrincipleComponent
from image_similarity import Embedding, cosine_similarity

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


if __name__ == "__main__":
    """Generates embeddings for all the images present in the given file path.
    """
    df_embeddings = Embedding(
        file_paths["input_images_folder_path"], file_paths
    ).get_embeddings()
    print(df_embeddings.head())
    print(cosine_similarity(df_embeddings[1], df_embeddings[2]))

    """Latent Representation of embeddings of the images.
    """
    df_embeddings_copy = df_embeddings.copy()
    df_embeddings.drop(labels=0, axis=1, inplace=True)
    emb = PrincipleComponent(df_embeddings, df_embeddings_copy).visualize_pca()

