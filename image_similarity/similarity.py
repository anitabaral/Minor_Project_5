
from scipy import spatial

def cosine_similarity(feature_vector_1, feature_vector_2):

    return 1 - spatial.distance.cosine(feature_vector_1, feature_vector_2)
