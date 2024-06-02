import numpy as np
import cv2
import os
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from hmmlearn import hmm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim

# Download and extract dataset
url = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
dataset_path = "att_faces.zip"
urllib.request.urlretrieve(url, dataset_path)
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(".")

# Load and resize images
image_dir = "./att_faces"
images = []
labels = []
for person in range(1, 41):
    for image_num in range(1, 11):
        img_path = os.path.join(image_dir, f"s{person}", f"{image_num}.pgm")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (92, 112))
        images.append(img_resized)
        labels.append(person)
images = np.array(images)
labels = np.array(labels)

# Compute average face
average_face = np.mean(images, axis=0)

# Show the average face
plt.imshow(average_face, cmap='gray')
plt.title("Average Face")
plt.show()

# Demean faces
demeaned_faces = images - average_face

# Show an example of a demeaned face
plt.imshow(demeaned_faces[0], cmap='gray')
plt.title("Demeaned Face")
plt.show()

# Flatten images and apply PCA
num_samples, height, width = images.shape
flattened_images = images.reshape(num_samples, height * width)
pca = PCA(n_components=100)
pca_features = pca.fit_transform(flattened_images)

# Calculate the covariance matrix of the PCA features
covariance_matrix = np.cov(pca_features, rowvar=False)
print("Covariance Matrix:\n", covariance_matrix)

# Train HMM for each subject
hmms = []
for person in range(1, 41):
    person_features = pca_features[labels == person]
    model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    model.fit(person_features)
    hmms.append(model)

# Function to recognize faces
def recognize_face(face):
    flattened_face = face.reshape(1, -1)
    pca_face = pca.transform(flattened_face)
    scores = [model.score(pca_face) for model in hmms]
    return np.argmax(scores) + 1

# Test recognition
test_face = images[0]
predicted_person = recognize_face(test_face)
print(f"Predicted Person: {predicted_person}")

# Take one of the PCA feature vectors
pca_vector = pca_features[0]

# Transform it back to the original space
inverse_pca = pca.inverse_transform(pca_vector)

# Reshape to the original image dimensions
reconstructed_image = inverse_pca.reshape((height, width))

# Display the reconstructed image
plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Image from PCA Vector")
plt.show()

# Compute Cosine Similarity between two PCA feature vectors
cos_sim = cosine_similarity([pca_features[0]], [pca_features[1]])
print(f"Cosine Similarity: {cos_sim[0][0]}")

# Compute Euclidean Distance between two PCA feature vectors
euc_dist = euclidean(pca_features[0], pca_features[1])
print(f"Euclidean Distance: {euc_dist}")

# Compute Pixel Similarity using Mean Squared Error (MSE)
def mean_squared_error(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

mse = mean_squared_error(images[0], images[1])
print(f"Mean Squared Error (Pixel Similarity): {mse}")

# Compute Pixel Similarity using Structural Similarity Index (SSIM)
ssim_index, _ = ssim(images[0], images[1], full=True)
print(f"Structural Similarity Index (SSIM): {ssim_index}")
