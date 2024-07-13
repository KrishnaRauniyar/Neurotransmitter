import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import argparse
import random
import numpy as np

def set_random_seeds(seed=32):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_random_seeds()

def kMeansClustering(cvsPath):
    df = pd.read_csv(cvsPath)

    # Parsing protein name into components (protein_id, chain_id, residue_number, residue_type)
    def parse_protein_name(protein_name):
        parts = protein_name.split('_')
        return parts[0], parts[1], parts[2], parts[3]

    df[['protein_id', 'chain_id', 'residue_type', 'residue_number']] = df['protein'].apply(lambda x: pd.Series(parse_protein_name(x)))

    # Dropping the original 'Protein Name' column
    df.drop(columns=['group', 'protein'], inplace=True)

    # Encoding categorical variables 
    label_encoders = {}
    for column in ['protein_id', 'chain_id', 'residue_type']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Defining features
    X = df.drop(columns=['residue_number']) 
    residue_types = df['residue_type']  

    # Standardizing the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Defining the autoencoder model usually important for dimensionality reduction
    input_dim = X_scaled.shape[1]
    encoding_dim = 32  # We can adjust this dimension

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)

    decoder = Dense(64, activation='relu')(encoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compiling the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32, validation_split=0.2)

    # Extracting the encoder part of the model to get the encoded representations
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    encoded_data = encoder_model.predict(X_scaled)

    # Applying k-means clustering
    n_clusters = len(label_encoders['residue_type'].classes_) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(encoded_data)

    # Adding cluster labels to the original dataframe
    df['cluster'] = clusters

    # Mapping cluster numbers to actual residue type names
    cluster_to_residue_type = {i: label_encoders['residue_type'].inverse_transform([i])[0] for i in range(n_clusters)}
    df['cluster_name'] = df['cluster'].map(cluster_to_residue_type)

    # Calculating ARI
    ari = adjusted_rand_score(df['residue_type'], df['cluster'])
    print(f'Adjusted Rand Index (ARI): {ari}')

    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded_data)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette('tab10', n_clusters) 
    sns.scatterplot(x='pca1', y='pca2', hue='cluster_name', data=df, palette=palette)
    plt.title('Clusters (PCA)')
    plt.legend(title='Residue Types', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("kmeans_PCA_plot.png", bbox_inches='tight')
    plt.close()

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(encoded_data)
    df['tsne1'] = tsne_result[:, 0]
    df['tsne2'] = tsne_result[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='tsne1', y='tsne2', hue='cluster_name', data=df, palette=palette)
    plt.title('Clusters (t-SNE)')
    plt.legend(title='Residue Types', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("kmeans_tSNE_plot.png", bbox_inches='tight')
    plt.close()

    # MDS visualization
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(encoded_data)
    df['mds1'] = mds_result[:, 0]
    df['mds2'] = mds_result[:, 1]

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette('tab10', n_clusters) 
    sns.scatterplot(x='mds1', y='mds2', hue='cluster_name', data=df, palette=palette)
    plt.title('Clusters (MDS)')
    plt.legend(title='Residue Types', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("kmeans_MDS_plot.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("k-Means Clustering")
    parser.add_argument('-p','--input_path', type=str, required=True, help="CSV input path")
    args = parser.parse_args()
    kMeansClustering(args.input_path)


