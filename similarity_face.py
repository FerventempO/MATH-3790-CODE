from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim


# Example: Compute cosine similarity between two PCA feature vectors
cos_sim = cosine_similarity([pca_features[0]], [pca_features[1]])
print(f"Cosine Similarity: {cos_sim[0][0]}")

# Example: Compute Euclidean distance between two PCA feature vectors
euc_dist = euclidean(pca_features[0], pca_features[1])
print(f"Euclidean Distance: {euc_dist}")
# Example: Compute pixel similarity using Mean Squared Error (MSE)
def mean_squared_error(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

mse = mean_squared_error(images[0], images[1])
print(f"Mean Squared Error (Pixel Similarity): {mse}")

# Example: Compute pixel similarity using Structural Similarity Index (SSIM)

ssim_index, _ = ssim(images[0], images[1], full=True)
print(f"Structural Similarity Index (SSIM): {ssim_index}")
