import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

# Paths to the dataset folders
train_path = '/Users/kriti/Downloads/Project_dataset-4/train/'
val_path = '/Users/kriti/Downloads/Project_dataset-4/val/'
labels_path = '/Users/kriti/Downloads/Project_dataset-4/val/labels.txt'

# Class mapping
classes = {
    'ई': 0, 'ऋ': 1, 'क़ै': 2, 'खु': 3, 'गो': 4, 'चा': 5, 'छः': 6, 'जा': 7,
    'झं': 8, 'ञ': 9, 'टी': 10, 'ढ़ी': 11, 'ण': 12, 'धौ': 13, 'ने': 14,
    'पं': 15, 'फ़': 16, 'बृ': 17, 'माँ': 18, 'रौ': 19, 'लॉ': 20, 'वैं': 21,
    'षु': 22, 'सृ': 23, 'हाँ': 24
}

# Valid extensions
valid_extensions = ('.jpg')

# Arrays to store data
X_train = []
Y_train = []
X_test = []
Y_test = []
val_images = []  # To track validation image names

# Preprocessing functions
def preprocess_image(img):
    """Reduce noise and resize image."""
    noise_free = cv2.GaussianBlur(img, (5, 5), 0)
    resized = cv2.resize(noise_free, (32, 32))  # Standardize to 32x32
    return resized.flatten()  # Convert to a 1D feature vector

# Load training data
train_image_names = set()
for cl, label in classes.items():
    train_cl_path = os.path.join(train_path, cl)
    if os.path.exists(train_cl_path):
        for img_name in os.listdir(train_cl_path):
            if img_name.endswith(valid_extensions):
                train_image_names.add(img_name)
                img_path = os.path.join(train_cl_path, img_name)
                img = cv2.imread(img_path, 0)  # Read in grayscale
                if img is not None:
                    X_train.append(preprocess_image(img))
                    Y_train.append(label)
        print(f"Processed training class: {cl}")
    else:
        print(f"Warning: Directory for class '{cl}' not found.")

# Load validation labels
val_labels = {}
with open(labels_path, 'r') as f:
    for line in f:
        try:
            img_name, label = line.strip().split('\t')
            if label in classes:
                val_labels[img_name] = classes[label]
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")

# Load validation data
for img_name in os.listdir(val_path):
    if img_name.endswith(valid_extensions):
        img_path = os.path.join(val_path, img_name)
        img = cv2.imread(img_path, 0)
        if img is not None:
            X_test.append(preprocess_image(img))
            Y_test.append(val_labels.get(img_name, -1))  # Default to -1 if label is missing
            val_images.append(img_name)

# Convert to numpy arrays
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
print("Training the SVM model...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, Y_train)
print("SVM model training completed.")

# Predict labels for the validation data
print("Predicting labels for validation data...")
Y_pred = svm_model.predict(X_test)

# Calculate overall accuracy
validation_accuracy = accuracy_score(Y_test, Y_pred)
print(f"Overall Validation Accuracy: {validation_accuracy:.2f}")

# Save the SVM model and scaler
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("SVM model and scaler saved.")

# Summary
print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(Y_test)}")


