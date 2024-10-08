#!/usr/bin/env python
# coding: utf-8

import os
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define the image size
IMAGE_SIZE = [224, 224]

# Paths to the train, test, and validation datasets
train_path = "dataset/train"
test_path = "dataset/test"
val_path = "dataset/val"

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:  # Check if the image is not empty
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return np.array(images)


# Load images for training, testing, and validation
x_train = load_images_from_folder(train_path)
x_test = load_images_from_folder(test_path)
x_val = load_images_from_folder(val_path)

# Normalize pixel values to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data augmentation for test and validation sets
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Flow validation and test images in batches of 32 using test_val_datagen generator
test_set = test_val_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

val_set = test_val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Define the VGG19 model
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the pre-trained layers
for layer in vgg.layers:
    layer.trainable = False

# Flatten the output of the VGG19 model
x = Flatten()(vgg.output)

# Add a dense layer with softmax activation for predictions
prediction = Dense(3, activation='softmax')(x)

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

# Fit the model
history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=5,
    shuffle=True
)

# Plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('vgg-loss-rps-1.png')
plt.show()

# Plot the accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('vgg-acc-rps-1.png')
plt.show()

# Evaluate the model on the test set
model.evaluate(test_set)

# Generate predictions for the test set
y_pred = model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_set.classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=training_set.class_indices.keys(), 
            yticklabels=training_set.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print(classification_report(test_set.classes, y_pred_classes))

# Save the model
model.save("vgg-rps-final.h5")