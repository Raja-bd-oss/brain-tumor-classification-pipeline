"""
SYSTÈME CNN DE CLASSIFICATION D'IMAGES MÉDICALES
(tumeur vs non tumeur)
Adapté pour fonctionner avec segment_region_growing.py
"""

# Importations principales
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import random

# Définition du modèle CNN (type LeNet amélioré)
def build_cnn(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)

    # Bloc 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc 2
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc 3
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couches fully connected
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


# ========================================================================
# PARAMÈTRES D'ENTRAÎNEMENT
# ========================================================================
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
IMG_SIZE = 128  # pour les images médicales
DATASET_PATH = r"C:\Users\User\Desktop\cv2\database"  # Dossier contenant "yes" et "no"

# ========================================================================
# PRÉPARATION DES DONNÉES
# ========================================================================
print("[INFO] Chargement des images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    data.append(image)

    # Label: 1 si tumeur, 0 sinon
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label.lower() in ["yes", "tumeur", "malign"] else 0
    labels.append(label)

# Conversion en tableaux numpy + normalisation
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Séparation train/test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# One-hot encoding
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Générateur d'augmentation
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ========================================================================
# ENTRAÎNEMENT DU MODÈLE
# ========================================================================
print("[INFO] Compilation du modèle...")
model = build_cnn(width=IMG_SIZE, height=IMG_SIZE, depth=3, classes=2)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Entraînement du modèle...")
start = time.time()
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1
)
end = time.time()

# ========================================================================
# SAUVEGARDE DU MODÈLE
# ========================================================================
print("[INFO] Sauvegarde du modèle entraîné...")
model.save("tumor_detection_model.h5")
print(f"✓ Modèle sauvegardé sous 'tumor_detection_model.h5'")
print(f"Durée d'entraînement: {(end - start):.2f} secondes")

# ========================================================================
# VISUALISATION DES RÉSULTATS
# ========================================================================
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="Train Acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Val Acc")
plt.title("Évolution de la Précision et de la Perte")
plt.xlabel("Époque")
plt.ylabel("Valeur")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
print("✓ Graphique d'entraînement sauvegardé sous 'training_plot.png'")
