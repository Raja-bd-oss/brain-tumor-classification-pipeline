import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.keras.backend.clear_session()
models = tf.keras.models
layers = tf.keras.layers
Input = tf.keras.layers.Input
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# ============================================================================
# GÉNÉRATION AUTOMATIQUE DE MASQUES
# ============================================================================

def generate_mask_automatically(img):
    """
    Génère automatiquement un masque de segmentation UNIQUEMENT pour la tumeur
    Détecte les régions anormalement claires (tumeurs) dans le cerveau
    
    Args:
        img: Image en niveaux de gris (0-255)
        
    Returns:
        Masque binaire (0-255) de la tumeur uniquement
    """
    # S'assurer que l'image est en uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_uint8)
    
    # Réduction du bruit
    denoised = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # STRATÉGIE 1: Seuillage sur les TRÈS hautes intensités (tumeur = zone la plus claire)
    # Calculer le percentile 90 pour isoler les zones les plus claires
    threshold_high = np.percentile(denoised, 90)
    _, high_intensity_mask = cv2.threshold(denoised, threshold_high, 255, cv2.THRESH_BINARY)
    
    # STRATÉGIE 2: Détecter les régions anormalement brillantes
    # Utiliser un seuil adaptatif pour trouver les anomalies locales
    adaptive_mask = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, -5
    )
    
    # Combiner les deux masques
    combined_mask = cv2.bitwise_and(high_intensity_mask, adaptive_mask)
    
    # Nettoyage morphologique
    kernel_small = np.ones((3,3), np.uint8)
    kernel_large = np.ones((5,5), np.uint8)
    
    # Enlever le bruit
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Combler les petits trous
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Trouver les contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Créer le masque final avec SEULEMENT la tumeur
    tumor_mask = np.zeros_like(img_uint8)
    
    if len(contours) > 0:
        # Filtrer les contours pour garder seulement la tumeur probable
        valid_contours = []
        img_area = img_uint8.shape[0] * img_uint8.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # La tumeur représente typiquement 5-30% de l'image
            if 0.05 * img_area < area < 0.35 * img_area:
                # Vérifier la circularité (tumeurs sont souvent arrondies)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    
                    # Garder les formes relativement compactes (circularity > 0.3)
                    if circularity > 0.3:
                        valid_contours.append(contour)
        
        # Si aucun contour valide trouvé, prendre le plus grand compact
        if len(valid_contours) == 0 and len(contours) > 0:
            # Trouver le contour le plus circulaire parmi les grands
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 0.03 * img_area:  # Au moins 3% de l'image
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        # Score = circularité * taille relative
                        score = circularity * (area / img_area)
                        if score > best_score:
                            best_score = score
                            best_contour = contour
            
            if best_contour is not None:
                valid_contours.append(best_contour)
        
        # Dessiner les contours valides
        if valid_contours:
            # Si plusieurs contours, prendre le plus grand
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(tumor_mask, [largest_contour], -1, 255, -1)
    
    return tumor_mask

# ============================================================================
# MODÈLE U-NET
# ============================================================================

def build_unet(input_shape=(128, 128, 1)):
    """Construit un modèle U-Net compact pour segmentation"""
    inputs = layers.Input(shape=input_shape)
    
    # Encodeur
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Fond
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    
    # Décodeur
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    return models.Model(inputs=[inputs], outputs=[outputs])

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice pour segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Perte Dice pour segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)

# ============================================================================
# CHARGEMENT DES DONNÉES AVEC GÉNÉRATION DE MASQUES
# ============================================================================

dataset_path = 'C:\\Users\\User\\Desktop\\cv2\\database'

data = []
labels = []
masks = []  # Pour stocker les masques générés automatiquement

IMG_SIZE = 128
categories = ['no','yes']

print("="*70)
print("🧠 CHARGEMENT DES DONNÉES + GÉNÉRATION AUTOMATIQUE DES MASQUES")
print("="*70)

for label, category in enumerate(categories):
    dataset = os.path.join(dataset_path, category)
    
    print(f"\n📂 Traitement du dossier '{category}'...")
    count = 0
    
    for i in os.listdir(dataset):
        img_path = os.path.join(dataset, i)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Générer le masque automatiquement
        if label == 1:  # Si c'est une image avec tumeur (yes)
            mask = generate_mask_automatically(img)
            print(f"   ✅ Masque généré pour {i}", end='\r')
        else:  # Si pas de tumeur (no)
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Normalisation
        img_normalized = img / 255.0
        mask_normalized = mask / 255.0
        
        data.append(img_normalized)
        masks.append(mask_normalized)
        labels.append(label)
        count += 1
    
    print(f"\n   ✅ {count} images traitées")

# Conversion en arrays
data = np.array(data, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print(f"\n✅ Dataset chargé:")
print(f"   - Images: {data.shape}")
print(f"   - Masques: {masks.shape}")
print(f"   - Labels: {labels.shape}")
print(f"   - Images 'no': {np.sum(labels == 0)}")
print(f"   - Images 'yes': {np.sum(labels == 1)}")

# Reshape pour les CNNs
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
masks = masks.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Split train/test pour DÉTECTION
x_train_det, x_test_det, y_train_det, y_test_det = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Split train/test pour SEGMENTATION (seulement les images avec tumeur)
tumor_indices = np.where(labels == 1)[0]
data_tumor = data[tumor_indices]
masks_tumor = masks[tumor_indices]

x_train_seg, x_test_seg, y_train_seg, y_test_seg = train_test_split(
    data_tumor, masks_tumor, test_size=0.2, random_state=42
)

print(f"\n📊 Split des données:")
print(f"   Détection - Train: {len(x_train_det)}, Test: {len(x_test_det)}")
print(f"   Segmentation - Train: {len(x_train_seg)}, Test: {len(x_test_seg)}")

# ============================================================================
# MODÈLE 1: DÉTECTION (VOTRE MODÈLE ORIGINAL)
# ============================================================================

print("\n" + "="*70)
print("🏗️ CONSTRUCTION DU MODÈLE DE DÉTECTION")
print("="*70)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train_det)

detection_model = models.Sequential([
    Input(shape=(128, 128, 1)),
    
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

detection_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print('✅ Modèle de détection prêt\n')

# Entraînement de la détection
print('🚀 Entraînement du modèle de détection...\n')

history_det = detection_model.fit(
    x_train_det, y_train_det, 
    batch_size=32, 
    epochs=20, 
    validation_data=(x_test_det, y_test_det),
    verbose=1
)

test_loss_det, test_acc_det = detection_model.evaluate(x_test_det, y_test_det, verbose=0)
print(f"\n✅ Détection - Test Accuracy: {test_acc_det:.2%}")

detection_model.save('tumor_detection_model.h5')
print("💾 Modèle de détection sauvegardé")

# ============================================================================
# MODÈLE 2: SEGMENTATION U-NET
# ============================================================================

print("\n" + "="*70)
print("🏗️ CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE U-NET")
print("="*70)

segmentation_model = build_unet(input_shape=(128, 128, 1))

segmentation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,
    metrics=[dice_coefficient, 'accuracy']
)

print('✅ Modèle U-Net prêt\n')

# Callbacks pour U-Net
callbacks_unet = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_unet_segmentation.h5',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# Entraînement de U-Net
print('🚀 Entraînement du modèle U-Net...\n')

history_seg = segmentation_model.fit(
    x_train_seg, y_train_seg,
    validation_data=(x_test_seg, y_test_seg),
    epochs=30,
    batch_size=16,
    callbacks=callbacks_unet,
    verbose=1
)

# Évaluation
test_results = segmentation_model.evaluate(x_test_seg, y_test_seg, verbose=0)
print(f"\n✅ Segmentation U-Net:")
print(f"   - Test Loss: {test_results[0]:.4f}")
print(f"   - Dice Coefficient: {test_results[1]:.4f}")
print(f"   - Accuracy: {test_results[2]:.4f}")

segmentation_model.save('unet_segmentation_trained.h5')
print(" Modèle U-Net sauvegardé")

# ============================================================================
# FONCTION COMPLÈTE DE PRÉDICTION
# ============================================================================

def analyze_brain_scan(detection_model, segmentation_model, image_path, visualize=True):
    """
    Analyse complète d'un scan cérébral
    1. Détection de tumeur
    2. Segmentation U-Net
    3. Extraction de caractéristiques
    
    Args:
        detection_model: Modèle de détection
        segmentation_model: Modèle U-Net
        image_path: Chemin vers l'image
        visualize: Afficher les résultats
        
    Returns:
        dict avec analyse complète
    """
    # Charger l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger: {image_path}")
    
    img_original = img.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img / 255.0
    img_input = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    print(f"\n{'='*70}")
    print(f"🔍 ANALYSE COMPLÈTE: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # ÉTAPE 1: Détection
    detection_prob = detection_model.predict(img_input, verbose=0)[0][0]
    has_tumor = detection_prob >= 0.5
    
    print(f"\n1️⃣  DÉTECTION:")
    print(f"   Résultat: {'✅ TUMEUR DÉTECTÉE' if has_tumor else '❌ PAS DE TUMEUR'}")
    print(f"   Probabilité: {detection_prob:.2%}")
    print(f"   Confiance: {(detection_prob if has_tumor else (1-detection_prob)):.2%}")
    
    results = {
        'image_path': image_path,
        'detection': {
            'has_tumor': bool(has_tumor),
            'probability': float(detection_prob),
            'confidence': float(detection_prob if has_tumor else (1-detection_prob))
        }
    }
    
    # ÉTAPE 2: Segmentation (si tumeur détectée)
    if has_tumor:
        print(f"\n2️⃣  SEGMENTATION U-NET:")
        
        # Prédiction du masque
        mask_pred = segmentation_model.predict(img_input, verbose=0)[0]
        mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
        mask_binary = mask_binary.squeeze()
        
        # Extraction des caractéristiques
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            print(f"   ✅ Segmentation réussie")
            print(f"\n3️⃣  CARACTÉRISTIQUES DE LA TUMEUR:")
            print(f"   📏 Surface: {int(area)} pixels²")
            print(f"   📏 Périmètre: {perimeter:.1f} pixels")
            print(f"   📍 Position: ({cx}, {cy})")
            print(f"   📐 Dimensions: {w} x {h} pixels")
            print(f"   🔵 Circularité: {circularity:.3f}")
            
            # Analyse de la forme
            if circularity > 0.8:
                shape_desc = "Très régulière (possiblement bénigne)"
            elif circularity > 0.6:
                shape_desc = "Modérément régulière"
            else:
                shape_desc = "Irrégulière (nécessite attention)"
            
            print(f"   🔬 Forme: {shape_desc}")
            
            results['segmentation'] = {
                'mask': mask_binary,
                'area': int(area),
                'perimeter': float(perimeter),
                'bounding_box': (int(x), int(y), int(w), int(h)),
                'centroid': (int(cx), int(cy)),
                'dimensions': (int(w), int(h)),
                'circularity': float(circularity),
                'shape_description': shape_desc
            }
        else:
            print("   ⚠️ Segmentation peu claire")
            results['segmentation'] = {'mask': mask_binary}
    else:
        results['segmentation'] = None
    
    # VISUALISATION
    if visualize:
        import matplotlib.pyplot as plt
        
        if has_tumor and results['segmentation'] and 'area' in results['segmentation']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Analyse Complète - {os.path.basename(image_path)}', 
                        fontsize=14, fontweight='bold')
            
            # Ligne 1
            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title('Image Originale')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(results['segmentation']['mask'], cmap='gray')
            axes[0, 1].set_title('Masque U-Net')
            axes[0, 1].axis('off')
            
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay_colored = overlay.copy()
            overlay_colored[results['segmentation']['mask'] > 0] = [0, 255, 0]
            overlay_final = cv2.addWeighted(overlay, 0.6, overlay_colored, 0.4, 0)
            axes[0, 2].imshow(cv2.cvtColor(overlay_final, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title('Overlay (Vert)')
            axes[0, 2].axis('off')
            
            # Ligne 2
            contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            x, y, w, h = results['segmentation']['bounding_box']
            cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = results['segmentation']['centroid']
            cv2.circle(contour_img, (cx, cy), 5, (255, 0, 0), -1)
            axes[1, 0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Contours et Centre')
            axes[1, 0].axis('off')
            
            # Histogramme de distribution
            axes[1, 1].hist(img.ravel(), bins=256, color='gray', alpha=0.7)
            axes[1, 1].set_title('Distribution des intensités')
            axes[1, 1].set_xlabel('Intensité')
            axes[1, 1].set_ylabel('Fréquence')
            
            # Informations textuelles
            axes[1, 2].axis('off')
            info_text = f"""
RÉSUMÉ DE L'ANALYSE:

Détection:
  • Probabilité: {detection_prob:.2%}
  • Confiance: {results['detection']['confidence']:.2%}

Segmentation:
  • Surface: {results['segmentation']['area']} px²
  • Périmètre: {results['segmentation']['perimeter']:.1f} px
  • Dimensions: {results['segmentation']['dimensions'][0]}x{results['segmentation']['dimensions'][1]}

Position:
  • Centre: {results['segmentation']['centroid']}
  
Forme:
  • Circularité: {results['segmentation']['circularity']:.3f}
  • {results['segmentation']['shape_description']}
            """
            axes[1, 2].text(0.1, 0.5, info_text, fontsize=9, 
                           verticalalignment='center', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f'Pas de tumeur détectée\nConfiance: {results["detection"]["confidence"]:.2%}', 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
            plt.show()
    
    print(f"{'='*70}\n")
    return results

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

print("\n" + "="*70)
print("💡 UTILISATION DU SYSTÈME COMPLET")
print("="*70)
print("""
Pour analyser une nouvelle image:

result = analyze_brain_scan(
    detection_model,
    segmentation_model,
    'C:\\\\Users\\\\User\\\\Desktop\\\\cv2\\\\database\\\\yes\\\\Y1.jpg',
    visualize=True
)
""")

print("\n✅ ENTRAÎNEMENT TERMINÉ!")
print(f"   📁 tumor_detection_model.h5 (Détection)")
print(f"   📁 unet_segmentation_trained.h5 (Segmentation U-Net entraîné)")
print(f"   📁 best_unet_segmentation.h5 (Meilleur modèle U-Net)")