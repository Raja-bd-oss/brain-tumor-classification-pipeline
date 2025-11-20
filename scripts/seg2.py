"""
SYSTÈME DE SEGMENTATION PAR CROISSANCE DE RÉGIONS + PRÉDICTION CNN
AVEC NETTOYAGE MORPHOLOGIQUE AVANCÉ
-------------------------------------------------------------------
- Étape 1 : Extraction du cerveau
- Étape 2 : Détection de graines et segmentation par croissance
- Étape 3 : NETTOYAGE du masque (suppression du bruit)
- Étape 4 : Application d'un modèle CNN pré-entraîné
- Étape 5 : Affichage et sauvegarde du résultat

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
from scipy import ndimage

# ============================================================
# ⚙️ PARAMÈTRES GLOBAUX
# ============================================================
IMG_SIZE = 128
MODEL_PATH = 'tumor_detection_model.h5'
RESULTS_DIR = 'results_region_growing'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paramètres de nettoyage morphologique (AJUSTABLES)
OPENING_KERNEL_SIZE = 7      # Augmenté de 5 à 7 pour supprimer plus de bruit
CLOSING_KERNEL_SIZE = 9      # Augmenté de 7 à 9 pour mieux remplir
MIN_TUMOR_AREA = 300         # Augmenté de 200 à 300 pour être plus strict

# ============================================================
# 🧩 CHARGEMENT DU MODÈLE CNN
# ============================================================
print("[INFO] Chargement du modèle CNN...")
print(f"[INFO] Chemin du modèle: {MODEL_PATH}")
print(f"[INFO] Le fichier existe? {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERREUR: Le fichier modèle '{MODEL_PATH}' n'existe pas!")
    print(f"📁 Répertoire actuel: {os.getcwd()}")
    print(f"📁 Fichiers dans le répertoire:")
    for f in os.listdir('.'):
        if f.endswith('.h5') or f.endswith('.keras'):
            print(f"   - {f}")
    print("\n💡 Solutions possibles:")
    print("   1. Vérifiez que le modèle est bien entraîné (train_model.py)")
    print("   2. Vérifiez le nom du fichier (tumor_detection_model.h5)")
    print("   3. Placez le modèle dans le même dossier que ce script")
    exit(1)

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès depuis '{MODEL_PATH}'")
    print(f"[INFO] Architecture du modèle:")
    model.summary()
except Exception as e:
    print(f"\n❌ ERREUR lors du chargement du modèle:")
    print(f"   Type d'erreur: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    print("\n💡 Causes possibles:")
    print("   1. Le fichier .h5 est corrompu")
    print("   2. Version de Keras/TensorFlow incompatible")
    print("   3. Le modèle n'a pas été sauvegardé correctement")
    import traceback
    traceback.print_exc()
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


def clean_tumor_mask(mask):
    """
    Nettoie le masque tumoral pour supprimer le bruit
    NETTOYAGE AGRESSIF pour éliminer tout le bruit
    
    Opérations :
    1. OUVERTURE FORTE : Supprime les petits objets isolés (bruit)
    2. FERMETURE : Remplit les petits trous dans la tumeur
    3. Suppression des composantes trop petites
    4. Conservation UNIQUEMENT de la plus grande composante
    5. Lissage final des contours
    
    Args:
        mask: Masque binaire brut
    
    Returns:
        cleaned_mask: Masque nettoyé
    """
    print("\n[NETTOYAGE] Application des opérations morphologiques...")
    
    # Vérifier que le masque n'est pas vide
    if np.sum(mask) == 0:
        print("⚠️  Masque vide, aucun nettoyage nécessaire")
        return mask
    
    pixels_avant = np.sum(mask > 0)
    print(f"  Pixels avant nettoyage: {pixels_avant}")
    
    # 1. OUVERTURE AGRESSIVE - Supprime tout le bruit
    # On fait plusieurs itérations avec un kernel de plus en plus grand
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=3)
    
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    mask_opened = cv2.morphologyEx(mask_opened, cv2.MORPH_OPEN, kernel_medium, iterations=2)
    
    pixels_apres_ouverture = np.sum(mask_opened > 0)
    bruit_supprime = pixels_avant - pixels_apres_ouverture
    print(f"  Après OUVERTURE: {pixels_apres_ouverture} pixels")
    print(f"  🧹 Bruit supprimé: {bruit_supprime} pixels ({bruit_supprime/pixels_avant*100:.1f}%)")
    
    # 2. Garder UNIQUEMENT la plus grande composante (avant fermeture)
    num_labels_before, labels_before, stats_before, _ = cv2.connectedComponentsWithStats(
        mask_opened, connectivity=8)
    
    if num_labels_before <= 1:  # Seulement le fond
        print("⚠️  Aucune région trouvée après ouverture")
        return np.zeros_like(mask)
    
    # Trouver la plus grande composante (ignorer le fond = 0)
    largest_label = 1 + np.argmax(stats_before[1:, cv2.CC_STAT_AREA])
    largest_area = stats_before[largest_label, cv2.CC_STAT_AREA]
    
    # Créer un masque avec UNIQUEMENT la plus grande composante
    mask_largest_only = np.zeros_like(mask)
    mask_largest_only[labels_before == largest_label] = 255
    
    print(f"  Composantes trouvées: {num_labels_before - 1}")
    print(f"  Gardée: composante #{largest_label} ({largest_area} pixels)")
    
    # 3. FERMETURE - Remplit les trous dans la tumeur
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE))
    mask_closed = cv2.morphologyEx(mask_largest_only, cv2.MORPH_CLOSE, 
                                   kernel_closing, iterations=3)
    
    pixels_apres_fermeture = np.sum(mask_closed > 0)
    print(f"  Après FERMETURE: {pixels_apres_fermeture} pixels")
    
    # 4. Remplir TOUS les trous internes
    mask_filled = ndimage.binary_fill_holes(mask_closed).astype(np.uint8) * 255
    
    pixels_apres_remplissage = np.sum(mask_filled > 0)
    print(f"  Après remplissage des trous: {pixels_apres_remplissage} pixels")
    
    # 5. Lissage final pour des contours propres
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
    
    # 6. Vérification finale de la taille
    final_pixels = np.sum(cleaned_mask > 0)
    
    if final_pixels < MIN_TUMOR_AREA:
        print(f"⚠️  Région finale trop petite ({final_pixels} < {MIN_TUMOR_AREA})")
        print("   Possible faux positif ou tumeur très petite")
    
    print(f"  ✅ Résultat final: {final_pixels} pixels")
    print(f"  📊 Réduction totale: {pixels_avant - final_pixels} pixels ({(pixels_avant - final_pixels)/pixels_avant*100:.1f}%)")
    
    return cleaned_mask


def segment_tumor_region_growing(image):
    """Pipeline complet de segmentation par croissance avec nettoyage."""
    brain_mask = extract_brain_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seeds = find_seed_points(gray)
    
    # Segmentation brute
    tumor_mask_raw = region_growing(gray, seeds)
    tumor_mask_raw = cv2.bitwise_and(brain_mask, tumor_mask_raw)
    
    # NETTOYAGE DU MASQUE
    tumor_mask_clean = clean_tumor_mask(tumor_mask_raw)
    
    return tumor_mask_clean, tumor_mask_raw, brain_mask, seeds


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


def display_and_save_results(img_original, tumor_mask_raw, tumor_mask_clean, label, confidence, output_path):
    """Afficher et sauvegarder le résultat avec comparaison avant/après nettoyage."""
    
    # Créer l'overlay avec le masque nettoyé
    overlay = cv2.addWeighted(img_original, 0.7, 
                              cv2.cvtColor(tumor_mask_clean, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.putText(overlay, f"{label} ({confidence*100:.2f}%)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                (0, 255, 0) if "PAS" in label else (0, 0, 255), 2)
    
    cv2.imwrite(output_path, overlay)
    print(f"🖼️ Résultat sauvegardé : {output_path}")

    # Affichage avec comparaison
    plt.figure(figsize=(16, 8))
    
    # Ligne 1 : Pipeline complet
    plt.subplot(2, 4, 1)
    plt.title("1. Image Originale", fontsize=12, fontweight='bold')
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("2. Masque Brut\n(avec bruit)", fontsize=12, fontweight='bold')
    plt.imshow(tumor_mask_raw, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("3. Masque Nettoyé\n(après ouverture)", fontsize=12, fontweight='bold')
    plt.imshow(tumor_mask_clean, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title(f"4. Résultat Final\n{label} ({confidence*100:.2f}%)", 
              fontsize=12, fontweight='bold',
              color='green' if "PAS" in label else 'red')
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Ligne 2 : Zoom sur les différences
    plt.subplot(2, 4, 5)
    plt.title("Bruit Détecté\n(masque brut - nettoyé)", fontsize=11)
    noise = cv2.subtract(tumor_mask_raw, tumor_mask_clean)
    plt.imshow(noise, cmap='Reds')
    plt.axis("off")
    noise_pixels = np.sum(noise > 0)
    plt.text(0.5, -0.1, f"{noise_pixels} pixels de bruit supprimés", 
             ha='center', transform=plt.gca().transAxes, fontsize=9, color='red')

    plt.subplot(2, 4, 6)
    plt.title("Contour de la Tumeur", fontsize=11)
    contour_img = img_original.copy()
    contours, _ = cv2.findContours(tumor_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title("Superposition\n(tumeur en rouge)", fontsize=11)
    overlay_transparent = img_original.copy()
    red_overlay = np.zeros_like(img_original)
    red_overlay[tumor_mask_clean > 0] = [0, 0, 255]
    overlay_transparent = cv2.addWeighted(overlay_transparent, 0.7, red_overlay, 0.3, 0)
    plt.imshow(cv2.cvtColor(overlay_transparent, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("Statistiques", fontsize=11)
    plt.axis("off")
    tumor_area = np.sum(tumor_mask_clean > 0)
    brain_area = np.sum(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY) > 20)
    tumor_percentage = (tumor_area / brain_area * 100) if brain_area > 0 else 0
    
    stats_text = f"""
    Détection: {label}
    Confiance: {confidence*100:.1f}%
    
    Aire tumorale: {tumor_area} px
    Aire cérébrale: {brain_area} px
    Taille relative: {tumor_percentage:.2f}%
    
    Bruit supprimé: {noise_pixels} px
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path.replace('.jpg', '_detailed.png').replace('.png', '_detailed.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 🚀 EXÉCUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    try:
        print("\n═══════════════════════════════════════════════════════════")
        print("🧠 SEGMENTATION + CLASSIFICATION DE TUMEURS CÉRÉBRALES")
        print("   AVEC NETTOYAGE MORPHOLOGIQUE AVANCÉ")
        print("═══════════════════════════════════════════════════════════")

        # Demander le chemin de l'image OU utiliser un chemin par défaut
        img_path = input("👉 Entrez le chemin de l'image médicale (ou ENTER pour test): ").strip()
        
        # Si vide, chercher une image de test
        if not img_path:
            test_paths = [
                r"C:\Users\User\Desktop\cv2\database\yes\Y1.jpg",
                r"C:\Users\User\Desktop\cv2\database\yes\Y1.png",
                r"C:\Users\User\Desktop\cv2\test_image.jpg",
            ]
            for path in test_paths:
                if os.path.exists(path):
                    img_path = path
                    print(f"✓ Utilisation de l'image de test: {img_path}")
                    break
        
        if not img_path or not os.path.exists(img_path):
            print("\n❌ Image introuvable.")
            print("💡 Vérifiez:")
            print("   1. Le chemin complet avec l'extension (.jpg, .png, etc.)")
            print("   2. Exemple: C:\\Users\\User\\Desktop\\cv2\\database\\yes\\Y1.jpg")
            
            # Lister les fichiers disponibles dans le dossier yes
            yes_folder = r"C:\Users\User\Desktop\cv2\database\yes"
            if os.path.exists(yes_folder):
                print(f"\n📁 Fichiers disponibles dans {yes_folder}:")
                for f in os.listdir(yes_folder)[:10]:  # Limiter à 10
                    print(f"   - {f}")
            exit(1)

        start_time = time.time()

        # Chargement et redimensionnement
        print("\n[CHARGEMENT] Lecture de l'image...")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Impossible de lire l'image: {img_path}")
            print("💡 Vérifiez que le fichier est une image valide (jpg, png, etc.)")
            exit(1)
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        print(f"✓ Image redimensionnée à {IMG_SIZE}x{IMG_SIZE}")

        # Étape 1 : Segmentation avec nettoyage
        print("\n[SEGMENTATION] Croissance de régions...")
        tumor_mask_clean, tumor_mask_raw, brain_mask, seeds = segment_tumor_region_growing(img_resized)
        print(f"✓ Seeds trouvés: {len(seeds)}")

        # Étape 2 : Préparation pour prédiction CNN
        print("\n[PRÉPARATION] Normalisation pour le CNN...")
        img_normalized = img_resized / 255.0
        img_normalized = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # Étape 3 : Prédiction
        print("\n[PRÉDICTION] Classification par CNN...")
        confidence, label = predict_tumor(model, img_normalized)
        print(f"\n{'='*60}")
        print(f"🎯 RÉSULTAT: {label}")
        print(f"📊 CONFIANCE: {confidence*100:.2f}%")
        print(f"{'='*60}")

        # Étape 4 : Sauvegarde et affichage
        filename = os.path.basename(img_path)
        output_path = os.path.join(RESULTS_DIR, f"result_clean_{filename}")
        display_and_save_results(img_resized, tumor_mask_raw, tumor_mask_clean, 
                                label, confidence, output_path)

        duration = time.time() - start_time
        print(f"\n⏱️  Durée totale : {duration:.2f} secondes")
        print("═══════════════════════════════════════════════════════════\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Programme interrompu par l'utilisateur (Ctrl+C)")
        exit(0)
    except Exception as e:
        print(f"\n\n❌ ERREUR CRITIQUE:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n📍 Trace complète:")
        import traceback
        traceback.print_exc()
        print("\n💡 Contactez le support avec cette erreur")
        exit(1)