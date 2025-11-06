"""
SYSTÈME DE SEGMENTATION PAR CROISSANCE DE RÉGIONS + PRÉDICTION CNN
-------------------------------------------------------------------
- Étape 1 : Extraction du cerveau
- Étape 2 : Détection de graines et segmentation par croissance
- Étape 3 : Application d’un modèle CNN pré-entraîné
- Étape 4 : Affichage et sauvegarde du résultat

"""

# ============================================================
# 📦 IMPORTATIONS
# ============================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import load_model
from keras.utils import img_to_array

# ============================================================
# ⚙️ PARAMÈTRES GLOBAUX
# ============================================================
IMG_SIZE = 128
MODEL_PATH = 'tumor_detection_model.h5'
RESULTS_DIR = 'results_region_growing'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 🧩 CHARGEMENT DU MODÈLE CNN
# ============================================================
print("[INFO] Chargement du modèle CNN...")
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Modèle chargé depuis '{MODEL_PATH}'")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    exit(1)


# ============================================================
# 🧠 FONCTIONS UTILITAIRES
# ============================================================

def extract_brain_mask(image):
    """Extraction du cerveau en supprimant le fond."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closing


def find_seed_points(mask):
    """Trouver les points de départ (graines) pour la croissance."""
    # On choisit les pixels les plus clairs comme points de départ
    bright_pixels = np.where(mask > 180)
    seeds = list(zip(bright_pixels[0], bright_pixels[1]))
    return seeds


def region_growing(image, seeds, threshold=10):
    """Segmentation par croissance de régions à partir de graines."""
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    for seed in seeds:
        x, y = seed
        if segmented[x, y] == 0:
            region_intensity = int(image[x, y])
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if segmented[cx, cy] == 0:
                    segmented[cx, cy] = 255
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < height and 0 <= ny < width:
                                if segmented[nx, ny] == 0 and abs(int(image[nx, ny]) - region_intensity) < threshold:
                                    stack.append((nx, ny))
    return segmented


def segment_tumor_region_growing(image):
    """Pipeline complet de segmentation par croissance."""
    brain_mask = extract_brain_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seeds = find_seed_points(gray)
    tumor_mask = region_growing(gray, seeds)
    tumor_mask = cv2.bitwise_and(brain_mask, tumor_mask)
    return tumor_mask, brain_mask, seeds


def predict_tumor(model, img_normalized):
    """Faire la prédiction CNN (binaire)"""
    prediction = model.predict(img_normalized, verbose=0)[0]
    if prediction.shape[0] == 1:  # cas sigmoid
        conf = float(prediction[0])
        label = "TUMEUR DÉTECTÉE" if conf > 0.5 else "PAS DE TUMEUR"
    else:  # cas softmax
        idx = np.argmax(prediction)
        conf = float(prediction[idx])
        label = "TUMEUR DÉTECTÉE" if idx == 1 else "PAS DE TUMEUR"
    return conf, label


def display_and_save_results(img_original, tumor_mask, label, confidence, output_path):
    """Afficher et sauvegarder le résultat."""
    overlay = cv2.addWeighted(img_original, 0.7, cv2.cvtColor(tumor_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.putText(overlay, f"{label} ({confidence*100:.2f}%)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "PAS" in label else (0, 0, 255), 2)
    cv2.imwrite(output_path, overlay)
    print(f"🖼️ Résultat sauvegardé : {output_path}")

    # Affichage
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("Image Originale")
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Masque Segmenté")
    plt.imshow(tumor_mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Résultat final\n{label} ({confidence*100:.2f}%)")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 🚀 EXÉCUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════════════════")
    print("🧠 SEGMENTATION + CLASSIFICATION DE TUMEURS CÉRÉBRALES")
    print("═══════════════════════════════════════════════════════════")

    img_path = input("👉 Entrez le chemin de l'image médicale : ").strip()

    if not os.path.exists(img_path):
        print("❌ Image introuvable. Vérifiez le chemin.")
        exit(1)

    start_time = time.time()

    # Chargement et redimensionnement
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Étape 1 : Segmentation
    tumor_mask, brain_mask, seeds = segment_tumor_region_growing(img_resized)

    # Étape 2 : Préparation pour prédiction CNN
    img_normalized = img_resized / 255.0
    img_normalized = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Étape 3 : Prédiction
    confidence, label = predict_tumor(model, img_normalized)
    print(f"\n[RESULTAT] {label} (confiance : {confidence*100:.2f}%)")

    # Étape 4 : Sauvegarde et affichage
    filename = os.path.basename(img_path)
    output_path = os.path.join(RESULTS_DIR, f"result_rg_{filename}")
    display_and_save_results(img_resized, tumor_mask, label, confidence, output_path)

    duration = time.time() - start_time
    print(f"\n Durée totale : {duration:.2f} secondes")
    print("═══════════════════════════════════════════════════════════")
