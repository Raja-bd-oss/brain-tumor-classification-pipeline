import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os  # Ajout de l'import manquant

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'tumor_detection_model.h5'
IMAGE_PATH = 'C:\\Users\\User\\Desktop\\cv2\\scripts\\non1.webp'  # ⭐ Changez ici

IMG_SIZE = 128

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

print("="*70)
print("🧠 DÉTECTION + LOCALISATION DE TUMEUR")
print("="*70)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Modèle chargé\n")

# ============================================================================
# FONCTION DE LOCALISATION DE LA TUMEUR
# ============================================================================

def locate_tumor(img):
    """
    Trouve la position de la tumeur dans l'image
    Utilise le traitement d'image pour détecter la zone la plus claire
    
    Args:
        img: Image en niveaux de gris (0-255)
        
    Returns:
        Coordonnées du rectangle (x, y, w, h) ou None
    """
    # S'assurer que l'image est en uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Réduction du bruit
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # Seuillage pour isoler les zones claires (tumeur)
    threshold = np.percentile(blurred, 85)  # 15% des pixels les plus clairs
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Nettoyage morphologique
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Filtrer les contours pour trouver la tumeur
    img_area = img.shape[0] * img.shape[1]
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # La tumeur représente généralement 5-40% de l'image
        if 0.05 * img_area < area < 0.4 * img_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                # Garder les formes compactes
                if circularity > 0.2:
                    valid_contours.append(contour)
    
    # Si aucun contour valide, prendre le plus grand
    if len(valid_contours) == 0:
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.03 * img_area:
                valid_contours = [largest]
    
    if len(valid_contours) == 0:
        return None
    
    # Prendre le plus grand contour valide
    tumor_contour = max(valid_contours, key=cv2.contourArea)
    
    # Obtenir le rectangle englobant
    x, y, w, h = cv2.boundingRect(tumor_contour)
    
    return (x, y, w, h)

# ============================================================================
# FONCTION DE TEST AVEC LOCALISATION
# ============================================================================

def detect_and_localize(model, image_path, show_steps=False):
    """
    Détecte si tumeur + localise et dessine un cadre
    
    Args:
        model: Modèle de détection
        image_path: Chemin vers l'image
        show_steps: Afficher les étapes intermédiaires
        
    Returns:
        dict avec résultats
    """
    # Charger l'image
    filename = os.path.basename(image_path)
    print(f"📂 Analyse de: {filename}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("❌ Impossible de charger l'image")
        return None
    
    # Prétraitement pour la détection
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    # ÉTAPE 1: Détection
    prediction = model.predict(img_input, verbose=0)[0][0]
    has_tumor = prediction >= 0.5
    confidence = prediction if has_tumor else (1 - prediction)
    
    print(f"\n{'='*70}")
    print(f"1️⃣  DÉTECTION:")
    print(f"   Résultat: {'✅ TUMEUR' if has_tumor else '❌ NORMAL'}")
    print(f"   Probabilité: {prediction:.2%}")
    print(f"   Confiance: {confidence:.2%}")
    
    result = {
        'has_tumor': has_tumor,
        'probability': prediction,
        'confidence': confidence,
        'bbox': None
    }
    
    # ÉTAPE 2: Localisation (si tumeur détectée)
    if has_tumor:
        print(f"\n2️⃣  LOCALISATION:")
        bbox = locate_tumor(img_resized)
        
        if bbox:
            x, y, w, h = bbox
            area = w * h
            print(f"   ✅ Tumeur localisée")
            print(f"   Position: ({x}, {y})")
            print(f"   Taille: {w}x{h} pixels ({area} px²)")
            result['bbox'] = bbox
        else:
            print(f"   ⚠️ Localisation difficile")
    
    print(f"{'='*70}\n")
    
    # VISUALISATION
    if has_tumor:
        if show_steps:
            # Affichage avec étapes intermédiaires
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            # Affichage simple
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Image originale
        ax_idx = 0
        axes[ax_idx].imshow(img_resized, cmap='gray')
        axes[ax_idx].set_title('Image Originale', fontsize=14, fontweight='bold')
        axes[ax_idx].axis('off')
        
        # Image avec cadre
        ax_idx = 1
        img_with_box = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        if bbox:
            x, y, w, h = bbox
            # Dessiner le rectangle rouge
            cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Dessiner le centre
            cx, cy = x + w//2, y + h//2
            cv2.circle(img_with_box, (cx, cy), 5, (255, 0, 0), -1)
            
            # Ajouter le texte
            text = f"Tumeur: {prediction:.1%}"
            cv2.putText(img_with_box, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        axes[ax_idx].imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
        axes[ax_idx].set_title(f'Tumeur Localisée ({prediction:.1%})', 
                              fontsize=14, fontweight='bold', color='red')
        axes[ax_idx].axis('off')
        
        # Étapes intermédiaires (si demandé)
        if show_steps and bbox:
            ax_idx = 2
            # Créer le masque binaire pour visualisation
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_resized)
            blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
            threshold = np.percentile(blurred, 85)
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
            
            axes[ax_idx].imshow(binary, cmap='gray')
            axes[ax_idx].set_title('Masque de Détection', fontsize=14, fontweight='bold')
            axes[ax_idx].axis('off')
        
        plt.suptitle(f'Analyse: {filename}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        # Affichage simple pour image normale
        plt.figure(figsize=(8, 8))
        plt.imshow(img_resized, cmap='gray')
        plt.title(f'Scan Normal\nConfiance: {confidence:.2%}', 
                 fontsize=14, fontweight='bold', color='green')
        plt.axis('off')
        plt.show()
    
    return result

# ============================================================================
# FONCTION POUR TESTER PLUSIEURS IMAGES
# ============================================================================

def test_multiple_images(model, image_paths):
    """Teste plusieurs images et affiche les résultats"""
    results = []
    
    for img_path in image_paths:
        result = detect_and_localize(model, img_path, show_steps=False)
        if result:
            results.append(result)
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*70)
    tumors_detected = sum(1 for r in results if r['has_tumor'])
    print(f"Total d'images testées: {len(results)}")
    print(f"Tumeurs détectées: {tumors_detected}")
    print(f"Scans normaux: {len(results) - tumors_detected}")
    localized = sum(1 for r in results if r.get('bbox') is not None)
    if tumors_detected > 0:
        print(f"Tumeurs localisées: {localized}/{tumors_detected}")
    print("="*70)
    
    return results

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    
    # Test 1: Une seule image avec étapes
    print("🔬 TEST 1: Analyse détaillée d'une image\n")
    result = detect_and_localize(model, IMAGE_PATH, show_steps=True)
    
    # Test 2: Plusieurs images (optionnel)
    print("\n" + "="*70)
    response = input("Voulez-vous tester plusieurs images? (o/n): ").strip().lower()
    
    if response == 'o':
        # Exemples d'images à tester
        images_to_test = [
            'C:\\Users\\User\\Desktop\\cv2\\database\\yes\\Y1.jpg',
            'C:\\Users\\User\\Desktop\\cv2\\database\\yes\\Y2.jpg',
            'C:\\Users\\User\\Desktop\\cv2\\database\\yes\\Y3.jpg',
            'C:\\Users\\User\\Desktop\\cv2\\database\\no\\No1.jpg',
        ]
        
        # Filtrer les images qui existent
        existing_images = [img for img in images_to_test if cv2.imread(img) is not None]
        
        if existing_images:
            print(f"\n🔬 TEST DE {len(existing_images)} IMAGES\n")
            results = test_multiple_images(model, existing_images)
        else:
            print("❌ Aucune image trouvée. Modifiez les chemins dans le script.")
    
    print("\n✅ Tests terminés!")
    print("\n💡 Pour tester une autre image:")
    print("   Modifiez la ligne 'IMAGE_PATH' en haut du script")