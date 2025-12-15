# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:58:21 2025

@author: WIN11
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception, VGG19, ResNet50, EfficientNetB0, DenseNet121, InceptionResNetV2, EfficientNetV2L
from tensorflow.keras.applications import MobileNet
#from tensorflow.keras.applications import ShuffleNetV2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet

from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionRV2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_EfficientNetV2L

import random

# Set seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_random_seeds(42)

# Path to your main data directory
# base_dir = 'C:/Users/win10/Desktop/Sufi-416/Luna22-ISMI/processed_data/processed_data/Texture'


#base_dir = 'E:/datasets/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
base_dir = 'E:/datasets/LC25000'
# base_dir ='C:/Users/win10/Desktop/Sufi-416/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
#img_size = (224, 224)
img_size = (200, 200)
# Get class labels from subfolder names
class_labels = sorted(os.listdir(base_dir))

# Function to load and preprocess images (500 per class)
def load_images_from_dir(directory, class_labels, max_per_class=561, model_variant='VGG16'):
    imgs = []
    lbls = []
    
    # Select the appropriate preprocess_input function based on model variant
    if model_variant == 'Xception':
        preprocess_input = preprocess_xception
    elif model_variant == 'VGG19':
        preprocess_input = preprocess_vgg19
    elif model_variant == 'ResNet50':
        preprocess_input = preprocess_resnet50
    elif model_variant == 'EfficientNetB0':
        preprocess_input = preprocess_efficientnet
    elif model_variant == 'DenseNet121':
        preprocess_input = preprocess_densenet
    elif model_variant == 'EfficientNetV2L':
        preprocess_input = preprocess_EfficientNetV2L
    elif model_variant == 'InceptionResNetV2':
        preprocess_input = preprocess_inceptionRV2
    elif model_variant == 'MobileNet':
            preprocess_input = preprocess_mobilenet                           
    else:
        raise ValueError("Invalid model variant.")

    for label in class_labels:
        class_path = os.path.join(directory, label)
        class_count = 0
        
        if os.path.isdir(class_path):
            # Get list of image files in this class
            img_files = os.listdir(class_path)
            random.shuffle(img_files)  # Shuffle to get random samples
            
            for img_file in img_files:
                if class_count >= max_per_class:
                    break
                    
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess_input(img.astype(np.float32))  # Apply the correct preprocessing
                    # img = img/255.0
                    imgs.append(img)
                    lbls.append(label)
                    class_count += 1
    
    return np.array(imgs), np.array(lbls)

# Load images (500 per class)
images, labels = load_images_from_dir(base_dir, class_labels, max_per_class=700, model_variant='MobileNet')

# Encode labels
le = LabelEncoder()
int_labels = le.fit_transform(labels)
labels = to_categorical(int_labels, num_classes=len(class_labels))

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Clear session to free up GPU memory
tf.keras.backend.clear_session()

# Image Size
#Image_Size = [224, 224]
Image_Size = [200, 200]
# Set the model variant (choose from: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0', 'DenseNet121')
MODEL_VARIANT = 'MobileNet'  # Options: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0', 'DenseNet121'

# Define the base model
base_model = None
if MODEL_VARIANT == 'Xception':
    base_model = Xception(
        input_shape=Image_Size + [3],
        weights=None,  # Set to 'imagenet' if you want to use pre-trained weights
        include_top=False,
        pooling=None
    )
elif MODEL_VARIANT == 'MobileNet':
    base_model = Xception(
        input_shape=Image_Size + [3],
        weights=None,  # Set to 'imagenet' if you want to use pre-trained weights
        include_top=False,
        pooling=None
    )    
elif MODEL_VARIANT == 'VGG19':
    base_model = VGG19(
        input_shape=Image_Size + [3],
        weights='imagenet',
        include_top=False,
        pooling=None
    )
elif MODEL_VARIANT == 'ResNet50':
    base_model = ResNet50(
        input_shape=Image_Size + [3],
        weights='imagenet',
        include_top=False,
        pooling=None
    )
elif MODEL_VARIANT == 'EfficientNetV2L':
    base_model = EfficientNetV2L(
        input_shape=Image_Size + [3],
        weights='imagenet',
        include_top=False,
        pooling=None
    )
elif MODEL_VARIANT == 'DenseNet121':
    base_model = DenseNet121(
        input_shape=Image_Size + [3],
        weights='imagenet',
        include_top=False,
        pooling=None
    )
elif MODEL_VARIANT == 'InceptionResNetV2':
    base_model = InceptionResNetV2(
        input_shape=Image_Size + [3],
        weights='imagenet',
        include_top=False,
        pooling=None
    )
else:
    raise ValueError("Invalid model variant. Choose from 'VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0', 'DenseNet121'")

# Freezing Layers - Layer Freezing Option
freeze_layers = True  # Set to True to freeze the base model layers
freeze_up_to =0  # Freeze up to layer index (if freeze_layers is True)

# Freeze the base model layers based on the option
for i, layer in enumerate(base_model.layers):
    if freeze_layers and i < freeze_up_to:
        layer.trainable = False
    else:
        layer.trainable = True

# # Freeze the base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# Build the model
model = Sequential()
model.add(base_model)

# Add classification head
model.add(GlobalAveragePooling2D())
# model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.5))
# model.add(Dense(23, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dropout(0.1))

model.add(Dense(5, activation='softmax'))  # Assuming 5 classes for classification

# Create the final model
model = Model(inputs=model.inputs, outputs=model.outputs)

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # You can adjust the learning rate
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']
)

TRAINING_EPOCHS = 50
BATCH_SIZE = 4

history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=TRAINING_EPOCHS,
                        initial_epoch=0)

                   
import matplotlib.pyplot as plt
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Model Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.legend()
plt.legend(loc='lower right')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.save('E:/research paper/LC25000 paper & results/on LC25000 dataset/SOTA models/LC25000/trained model and Epoch CSV/MobileNet.h5')


#model.save('E:/research paper/LC25000 paper & results/on IQ-OTHNCCD dataset/Trained and epoch CSV/MobileNet.h5')
from tensorflow.keras.models import load_model
import tensorflow as tf
model=tf.keras.models.load_model('E:/research paper/LC25000 paper & results/Trained model/model-LC25000.h5')


model_name=""
# epochs=epochs
import csv
with open('E:/research paper/LC25000 paper & results/on LC25000 dataset/SOTA models/LC25000/trained model and Epoch CSV/MobileNet.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if the file is empty to write the header
        file_empty = file.tell() == 0
        if file_empty:
            header = ['Epoch']
            for model_name in [model_name]:
                header.append(f'{model_name} Train Accuracy')
                header.append(f'{model_name} Validation Accuracy')
                header.append(f'{model_name} Train Loss')
                header.append(f'{model_name} Validation Loss')
            writer.writerow(header)
        
        # Write the results for each epoch
        for epoch in range(50):
            row = [epoch + 1]  # Epoch number (1-indexed)
            
            # Collect the metrics for this epoch
            train_accuracy = history.history['accuracy'][epoch]
            val_accuracy = history.history['val_accuracy'][epoch]
            train_loss = history.history['loss'][epoch]
            val_loss = history.history['val_loss'][epoch]
            
            # Append the data to the row for the current epoch
            row.append(train_accuracy)
            row.append(val_accuracy)
            row.append(train_loss)
            row.append(val_loss)
            
            # Write the row of data to the file
            writer.writerow(row)

print(f"{model_name} training and validation data saved to 'models_training_and_validation_data.csv'.")

print(class_labels)


# Predict validation data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

#-----------to store these values-----
import pandas as pd
import numpy as np

# Predict validation data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Create a DataFrame to store the predictions and true values
df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'y_pred_prob_0': y_pred_probs[:, 0],  # Probability for class 0 (if you have more classes, add accordingly)
    'y_pred_prob_1': y_pred_probs[:, 1],  # Probability for class 1
    # Add more columns for other classes if needed
})

# Save to CSV
df.to_csv('E:/research paper/LC25000 paper & results/on IQ-OTHNCCD dataset/FPR TPR CSV/MobileNet.csv', index=False)

#-------------------end ---------------


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
# Define class labels in the correct order
# class_names = ["ACC", "LCC", "Normal", "SCC"]
# class_names = ["Colon-aca",  "Colon-n", "Lung-aca", "Lung-n", "Lung-scc"]
class_names = ["Benign","Malignant", "Normal"]

# Confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
#import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_true, y_pred))


# Plot with seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names ,annot_kws={"size": 18, "weight": "bold"})
plt.xlabel('Predicted Label', fontsize=18, weight='bold')
plt.ylabel('True Label', fontsize=18, weight='bold')
# plt.title('Confusion Matrix', fontsize=14)
plt.xticks(fontsize=15, weight='bold', rotation=45)
plt.yticks(fontsize=15, weight='bold', rotation=45)
plt.tight_layout()
plt.show()


# ROC-AUC calculation and plotting
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

n_classes = y_test.shape[1]
# class_labels=2
# y_true1 = to_categorical(y_true, num_classes=2)

y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']  # Adjust if number of classes differs
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')



plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('Multi-class ROC Curve', fontsize=16)

# plt.legend(loc="lower right", fontsize=16, ncol=1, frameon=True)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor=(0.99, 0.01), borderaxespad=0.)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

roc_auc_macro = roc_auc_score(y_true_bin, y_pred_probs, average='macro')
print(f"Macro-average ROC-AUC: {roc_auc_macro:.4f}")

#----------------FPS and prediction time---------------
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

# --- Prediction Function for FPS ---
def predict_fn(images):
    images = preprocess_input(images.astype(np.float32))
    return model.predict(images)

# --- Measure Prediction Time and FPS ---
def measure_prediction_time_and_fps(img_path, num_runs=100):
    # Load the image
    img_input, img_display = prepare_image(img_path)

    # Start measuring time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(img_input)
    end_time = time.time()

    # Total time for predictions
    total_time = end_time - start_time

    # Calculate FPS
    fps = num_runs / total_time

    # Time per prediction
    avg_prediction_time = total_time / num_runs

    return avg_prediction_time, fps

# --- Estimating GFLOPs ---
def calculate_fps_and_flops(model):
    # Initialize FLOPs counter
    total_flops = 0

    # Conv2D Layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Extract relevant parameters
            input_shape = layer.input.shape[1:]  # Exclude batch size
            output_shape = layer.output.shape[1:]
            kernel_size = layer.kernel_size[0]
            input_channels = input_shape[-1]
            output_channels = output_shape[-1]

            # Calculate FLOPs for Conv2D
            flops_per_conv = np.prod(output_shape) * input_channels * output_channels * kernel_size ** 2
            total_flops += flops_per_conv

        # Depthwise Conv2D
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            input_shape = layer.input.shape[1:]  # Exclude batch size
            output_shape = layer.output.shape[1:]
            kernel_size = layer.kernel_size[0]
            input_channels = input_shape[-1]
            output_channels = output_shape[-1]

            # Calculate FLOPs for DepthwiseConv2D
            flops_per_depthwise = np.prod(output_shape) * input_channels * kernel_size ** 2
            total_flops += flops_per_depthwise

        # Dense Layers
        elif isinstance(layer, tf.keras.layers.Dense):
            input_units = layer.input.shape[-1]
            output_units = layer.output.shape[-1]

            # Calculate FLOPs for Dense layer
            flops_per_dense = 2 * input_units * output_units
            total_flops += flops_per_dense

    # Convert total FLOPs to GFLOPs (1e9 FLOPs = 1 GFLOP)
    gflops = total_flops / 1e9
    return gflops

# Calculate GFLOPs for the model
gflops = calculate_fps_and_flops(model)
print(f"Estimated GFLOPs: {gflops:.2f} GFLOPs")

# --- Prepare single image ---
def prepare_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img_resized  # img_resized is in [0,255] RGB

# Usage example:
# img_path = 'C:/Users/win10/Desktop/Sufi-416/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Malignant cases/Malignant case (2).jpg'  # <-- change to your test image

img_path ='E:/datasets/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Malignant cases/Malignant case (1).jpg'
avg_pred_time, fps = measure_prediction_time_and_fps(img_path, num_runs=100)
print(f"Average prediction time per image: {avg_pred_time*1000:.2f} ms")
print(f"Frames per second (FPS): {fps:.2f}")
#----------------------end---------------


#---------------------precision and AUC---------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# Assuming y_pred_probs and y_true are already defined as your model's predictions and true labels
# y_pred_probs: Probability predictions of shape (n_samples, n_classes)
# y_true: True labels (not one-hot encoded), shape (n_samples,)

# Binarize the true labels for multi-class tasks
y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3, 4])  # Adjust for your classes

# Number of classes
n_classes = y_true_binarized.shape[1]

# Initialize arrays for precision, recall, and AUC
precision = dict()
recall = dict()
pr_auc = dict()

# Compute Precision-Recall curve and AUC for each class
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_probs[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Calculate micro-average Precision-Recall curve
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), y_pred_probs.ravel())
pr_auc["micro"] = auc(recall["micro"], precision["micro"])

# Plot Precision-Recall curve for each class and the micro-average curve
plt.figure(figsize=(10, 6))

# Plot PR curve for each class
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.2f})')

# Plot micro-average PR curve
plt.plot(recall["micro"], precision["micro"], label=f'Micro-average (AUC = {pr_auc["micro"]:.2f})', color='black', linestyle='--')

# Labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Multi-Class)')
plt.legend(loc='lower left')

# Show the plot
plt.show()

# Print out AUC for each class and micro-average AUC
for i in range(n_classes):
    print(f'Class {i} AUC: {pr_auc[i]:.4f}')
print(f'Micro-average AUC: {pr_auc["micro"]:.4f}')

# Optionally calculate ROC AUC for comparison (using one-vs-rest strategy)
roc_auc = roc_auc_score(y_true_binarized, y_pred_probs, average="macro", multi_class="ovr")
print(f'Macro-average ROC AUC: {roc_auc:.4f}')



#-------------end-------------

import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Define class labels in the correct order
#class_names = ["ACC", "LCC", "Normal", "SCC"]
# class_names = ["Adenocarcinoma", "Large Cell", "Normal", "Squamous Cell"]
class_names = ["Benign","Malignant",  "Normal"]

# Prepare the input image
def prepare_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)                           # Read image (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Convert to RGB
    img = cv2.resize(img, target_size)                   # Resize to 224x224
    img_array = image.img_to_array(img)                  # To float32 array
    img_array = np.expand_dims(img_array, axis=0)        # Add batch dimension
    img_array = preprocess_input(img_array)              # VGG16 preprocessing
    return img_array, img

# Predict and show
def predict_and_show(img_path):
    img_input, img_display = prepare_image(img_path)
    prediction = model.predict(img_input, verbose=0)     # Use existing model
    class_idx = np.argmax(prediction)
    label = class_names[class_idx]
    confidence = prediction[0][class_idx]

    # Display
    plt.imshow(img_display)
    plt.title(f"Prediction: {label} ({confidence:.2%})", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# # Example usage (no model reload)
# img_path = 'E:/research paper/VGG-ConvNeXT paper/new model with good results/results/orignal images and there predictions/orignal/ACC'
# predict_and_show(img_path)

def predict_and_show_all_images(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, fname)
            print(f"Processing: {fname}")
            predict_and_show(img_path)


# === Example usage ===
folder_path = r'E:/research paper/VGG-ConvNeXT paper/visual results comparison/orginal'
predict_and_show_all_images(folder_path)



