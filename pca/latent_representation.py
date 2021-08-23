import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PrincipleComponent:
    def __init__(self, csv, csv_copy):
        self.csv = csv
        self.csv_copy = csv_copy

    def two_principle_component(self):
        """Using PCA to reduce the dimensionality for latent representation.

        Returns:
            object: Dataframe consisting two principle components as columns.
        """
        scalar = StandardScaler().fit_transform(self.csv)
        pca_ = PCA(n_components=2)
        principal_components = pca_.fit_transform(scalar)
        principal_components_df = pd.DataFrame(
            data=principal_components,
            columns=["Principal component 1", "Principal component 2"],
        )

        return principal_components_df

    def visualize_pca(self):
        """Visualing the pricinple components of images on 2D scatter plot."""
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Principal Component - 1", fontsize=17)
        plt.ylabel("Principal Component - 2", fontsize=17)
        plt.title("Principal Component Analysis of flower Dataset", fontsize=18, pad=15)
        targets = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
        colors = ["b", "g", "r", "y", "m"]
        for target, color in zip(targets, colors):
            indices_to_keep = self.csv_copy[0] == target
            plt.scatter(
                self.two_principle_component().loc[
                    indices_to_keep, "Principal component 1"
                ],
                self.two_principle_component().loc[
                    indices_to_keep, "Principal component 2"
                ],
                c=color,
                s=40,
            )

        plt.legend(targets, prop={"size": 15})
        plt.savefig("latent_representation.png")
        print(
            "Principle components saved as latent_representation.png in the root directory"
        )
