import yaml

from pca import PrincipleComponent
from image_similarity import Embedding

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)

if __name__ == "__main__":
    # Embeddings
    df_embeddings = Embedding(
        file_paths["input_images_folder_path"], file_paths
    ).get_embeddings()
    print(df_embeddings.head())

    # PCA
    df_embeddings_copy = df_embeddings.copy()
    df_embeddings.drop(labels=0, axis=1, inplace=True)
    emb = PrincipleComponent(df_embeddings, df_embeddings_copy).visualize_pca()
