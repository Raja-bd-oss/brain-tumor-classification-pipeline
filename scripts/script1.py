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

dataset_path = 'C:\\Users\\User\\Desktop\\cv2\\database'

data = []
labels = [] 

IMG_SIZE = 128

categories = ['no','yes']

for label , category in enumerate(categories):      #label to encode no as 0 and yes as 1
    dataset = os.path.join(dataset_path,category)    
    for i in os.listdir(dataset):       #inside folders yes and no 
        img = cv2.imread(os.path.join(dataset,i),cv2.IMREAD_GRAYSCALE) #read image in grayscale

        if img is None:
            continue
        
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))   #resize to (128,128)

        img = img / 255.0   #normalise all images 

        data.append(img)        #append pixel range into data list

        labels.append(label)    #append label 0/1 into labels list

data = np.array(data, dtype=np.float32) #convert to numpy arrays since tensorflow expects data in array format 
labels = np.array(labels, dtype=np.int32) 

print(data.shape)

data = data.reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_train ,x_test , y_train , y_test = train_test_split(data,labels,test_size=0.2 , random_state= 42)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

model = models.Sequential([
    Input(shape=(128, 128, 1)),
    
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    Dropout(0.3),  # reduced dropout
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

print('Training Started...')

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))



print('Training Finished')

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")