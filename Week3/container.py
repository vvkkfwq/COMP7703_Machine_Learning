import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml


def perform_pca(data, n_components=2):
    """
    Perform principal component analysis (PCA) on the data

    parameter:
    data (DataFrame): input data
    n_components (int): Number of principal components

    output:
    DataFrame: DataFrame with principal components
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    principal_df = pd.DataFrame(
        data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return principal_df, pca


# Load the swissroll dataset
swissroll = np.loadtxt("./swissroll.txt")

# Perform KMeans clustering to generate class labels
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(swissroll)

# Perform PCA
pca_result, pca = perform_pca(swissroll, n_components=3)
print(pca_result)

# Plot the data in the space spanned by the first two principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    pca_result["PC1"], pca_result["PC2"], c=labels, cmap="viridis", alpha=0.5
)
plt.colorbar(scatter, label="Class")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Swissroll Data in PCA Space")
plt.show()

# Calculate the variance explained by the first two principal components
explained_variance = pca.explained_variance_ratio_
variance_percentage = explained_variance[:2].sum() * 100
print(
    f"Percentage of variance explained by the first two principal components: {variance_percentage:.2f}%"
)


# Plot Scree graph
plt.figure(figsize=(10, 7))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o",
    linestyle="--",
)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")
plt.show()
