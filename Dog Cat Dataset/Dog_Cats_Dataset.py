# Dog / Cats Dataset

import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset_dir = 'dataset'
cat_dir = os.path.join(dataset_dir, 'cats')
dog_dir = os.path.join(dataset_dir, 'dogs')

# Load images 
def load_images_and_labels(cat_dir, dog_dir, img_size=(128,128)):
    images = []
    labels = []

    # Cats
    for filename in os.listdir(cat_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(cat_dir, filename))
            if img is not None:
                images.append(cv2.resize(img, img_size))
                labels.append(0)

    # Dogs
    for filename in os.listdir(dog_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(dog_dir, filename))
            if img is not None:
                images.append(cv2.resize(img, img_size))
                labels.append(1)

    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

images, labels = load_images_and_labels(cat_dir, dog_dir)
print("Total Images:", len(images))

features = extract_hog_features(images)
print("Feature shape:", features.shape)

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    features, labels, images, test_size=0.5, random_state=42
)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,4))
for i in range(len(img_test)):
    plt.subplot(1, len(img_test), i+1)
    plt.imshow(cv2.cvtColor(img_test[i], cv2.COLOR_BGR2RGB))
    plt.title("Dog" if y_pred[i]==1 else "Cat")
    plt.axis("off")
plt.show()