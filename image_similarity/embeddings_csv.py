
import pandas as pd

from .embeddings import Embeddings

class EmbeddingsCsv:

  def ___init__(self):
    self.embeddings = Embeddings.get_embeddings()

  def get_embeddings_dataframe(self):

    total_embeddings_df = pd.DataFrame(columns=(np.arange(1, 4097)))
    for index, value in enumerate(self.embeddings):
      total_embeddings_df.loc[index] = value[0]
    total_embeddings_df.to_csv('/content/Minor_project_5/data/embeddings.csv')

    return total_embeddings_df

EmbeddingsCsv.get_embeddings_dataframe()