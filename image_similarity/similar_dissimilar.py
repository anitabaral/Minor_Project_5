import random
from scipy import spatial


def cosine_similar(df):
    """Finds the similarity between two feature vectors

    Args:
        df (dataframe): dataframe of the embeddings.
    """
    feature_vector_1 = df[random.randint(0, 9)].to_numpy()
    feature_vector_2 = df[random.randint(0, 9)].to_numpy()
    return 1 - spatial.distance.cosine(feature_vector_1, feature_vector_2)


def euclidean_dissimilar(df):
    """Finds the dissimilarity between two feature vectors

    Args:
        df (dataframe): dataframe of the embeddings.
    """
    feature_vector_1 = df[random.randint(0, 9)].to_numpy()
    feature_vector_2 = df[random.randint(0, 9)].to_numpy()

    return spatial.distance.euclidean(feature_vector_1, feature_vector_2)
