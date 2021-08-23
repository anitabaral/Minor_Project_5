# Minor_Project_5

The Objective of this project is to calculate the image similarity with embeddings using VGG16 pre-trained model and create a latent representation for the embeddings after PCA.

### Folder Description:
#### 1. Notebook
- Contains create_model.ipnyb which is used to create a model of our own. A detailed description is provided within the notebook file.

#### 2. Image Similarity
- Three files namely models.py(loads the model), images.py(loads the input images) and embeddings.py(creates embeddings for the input images)

#### 3. pca
- Contains latent_representation.py(deduces the features vectors to two principle components and plots it in a graph).


### How to run this repository:
Pre-requisits: Install pipenv(sudo apt-get pipenv)
#### Step 1: Setup
- Clone the repo
- pipenv shell

#### Step 2: Setup folders
- create a dataset folder
- Add subfolder named flowers_similarity
- Add images of flowers. For ease here is a link: https://drive.google.com/drive/folders/1GnqUyEzItXDO3bXxV2lBYZfO-YJKdU6J?usp=sharing

- create a model folder
- You can download the model from this link if you don't want to run create_model.ipnyb : https://drive.google.com/drive/folders/1-2Jcaeiu7yqPxkEIwDHVBXgCO1DQ9hYq?usp=sharing

#### Step 3: Run
- run the code "python app.py"

#### Output:
- In the output, you will see a dataframe consisting of labels of flowers and it's corresponding embeddings.
- The similarity and dissimilarity score.
- The latent representation after PCA is saved in the main folder as "latent_representation.png".
