import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class PrincipleComponent:
    def __init__(self, csv, csv_copy):
        self.csv = csv
        self.csv_copy = csv_copy

    def two_principle_component(self):
        scalar = StandardScaler().fit_transform(self.csv)
        pca_ = PCA(n_components=2)
        principal_components = pca_.fit_transform(scalar)
        principal_components_df = pd.DataFrame(
            data=principal_components,
            columns=['Principal component 1', 'Principal component 2'])
        return principal_components_df

    def visualize_pca(self):
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Principal Component - 1', fontsize=17)
        plt.ylabel('Principal Component - 2', fontsize=17)
        plt.title("Principal Component Analysis of flower Dataset", fontsize=18, pad=15)
        targets = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        colors = ['r', 'g', 'b', 'y', 'w']
        for target, color in zip(targets, colors):
            indicesToKeep = self.csv_copy.label == target
            plt.scatter(self.two_principle_component().loc[indicesToKeep, 'Principal component 1']
                        , self.two_principle_component().loc[indicesToKeep, 'Principal component 2'], c=color, s=40)

        plt.legend(targets, prop={'size': 15})
        plt.savefig('fig.png')


csv = pd.read_csv('../csv/embeddings.csv')
csv_copy = csv.copy()
csv.drop(labels="Unnamed: 0", axis=1, inplace=True)
csv.drop(labels="label", axis=1, inplace=True)
emb = PrincipleComponent(csv, csv_copy)
emb.visualize_pca()
