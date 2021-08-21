import pandas as pd

from .embeddings import Embedding

class FeaturesCsv:

  def ___init__(self, path):
    self.csv_path = path
    self.embeddings = Embedding.get_embeddings()

  def get_embeddings_dataframe(self):

    total_embeddings_df = pd.DataFrame(columns=(np.arange(1, 4097)))
    for index, value in enumerate(self.embeddings):
      total_embeddings_df.loc[index] = value[0]
    total_embeddings_df.to_csv(self.csv_path)

    return total_embeddings_df
