import os
import cv2
import mtcnn
import numpy as np
import time
from datetime import timedelta
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
# Constants
FACE_DATA = 'Faces/'
REQUIRED_SHAPE = (250, 250)
EPOCHS = 10
BATCH_SIZE = 32
# Initialize face detector
face_detector = mtcnn.MTCNN()
# Normalize function
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std
# Custom callback for logging
class LoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        time_elapsed = time.time() - self.epoch_time_start
        print(f"Epoch {epoch+1}/{EPOCHS} - Time: {timedelta(seconds=int(time_elapsed))}")
        print(f"Train Accuracy: {logs['accuracy']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")
        print(f"Train Loss: {logs['loss']:.4f}, Validation Loss: {logs['val_loss']:.4f}")
        print("-" * 50)
# Load and preprocess data
print("Starting data preprocessing...")
start_time = time.time()
X = []
y = []
label_dict = {}
total_images = sum([len(files) for r, d, files in os.walk(FACE_DATA)])
processed_images = 0
for i, face_name in enumerate(os.listdir(FACE_DATA)):
    person_dir = os.path.join(FACE_DATA, face_name)
    print("person's directory",person_dir)
    label_dict[i] = face_name
    print("current Face Name", face_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        x = face_detector.detect_faces(img_RGB)
        if not x:
            continue
        x1, y1, width, height = x[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1+width, y1+height
        face = img_RGB[y1:y2, x1:x2]
        face = normalize(face)
        face = cv2.resize(face, REQUIRED_SHAPE)
        X.append(face)
        y.append(i)
        processed_images += 1
        if processed_images % 100 == 0:
            print(f"Processed {processed_images}/{total_images} images")
X = np.array(X)
y = np.array(y)
y = to_categorical(y)
preprocessing_time = time.time() - start_time
print(f"Data preprocessing completed in {timedelta(seconds=int(preprocessing_time))}")
print(f"Total images processed: {len(X)}")
print(f"Number of classes: {len(label_dict)}")
print("-" * 50)
# Create the model
print("Creating and compiling the model...")
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(len(label_dict), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
print("Starting model training...")
training_start_time = time.time()
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[LoggingCallback()])
training_time = time.time() - training_start_time
print(f"Training completed in {timedelta(seconds=int(training_time))}")
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
# Save the model
print("Saving the model...")
model.save('face_recognition_model.h5')
# Save label dictionary
import pickle
with open('label_dict.pkl', 'wb') as f:
    pickle.dump(label_dict, f)
print("Training completed and model saved.")
print(f"Total time: {timedelta(seconds=int(time.time() - start_time))}")