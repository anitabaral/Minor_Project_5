import pandas as pd

from .embeddings import Embedding

class FeaturesCsv:

  def ___init__(self, path):
    self.csv_path = path
    self.embeddings = Embedding.get_embeddings()

  def get_embeddings_dataframe(self):

    total_embeddings_df = pd.DataFrame(columns=(np.arange(1, 4097)))
    for key, values in self.embeddings.items():
      for index, value in enumerate(values):
        value_list = np.ndarray.tolist(value)
        value_list[0].insert(0, key)
        total_embeddings_df.loc[len(total_embeddings_df)] = value_list[0]
    total_embeddings_df.to_csv(self.csv_path)
    
    return total_embeddings_df
