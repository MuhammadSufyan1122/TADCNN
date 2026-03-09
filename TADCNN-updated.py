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
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionRV2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_EfficientNetV2L

import random
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    ReLU,
    Add,
    Multiply,
    Concatenate,
    Softmax,
    Activation,
    Reshape
)
# Set seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_random_seeds(42)

base_dir = 'E:/datasets/LC25000'
img_size = (224, 224)

# Get class labels from subfolder names
class_labels = sorted(os.listdir(base_dir))

# Function to load and preprocess images (500 per class)
def load_images_from_dir(directory, class_labels, max_per_class=5000):
    imgs = []
    lbls = []
    
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
                    img_array = img_to_array(img) / 255.0
                    img = preprocess_input(img.astype(np.float32))
                    imgs.append(img)
                    lbls.append(label)
                    class_count += 1
    
    return np.array(imgs), np.array(lbls)

# Load images (500 per class)
images, labels = load_images_from_dir(base_dir, class_labels, max_per_class=5000)

# Encode labels
le = LabelEncoder()
int_labels = le.fit_transform(labels)
labels = to_categorical(int_labels, num_classes=len(class_labels))
# labels = tf.one_hot (labels, depth=len(class_labels))
print(labels.shape)



# Split data
X_train1, X_test1, y_train1, y_test1 = train_test_split(images, labels, test_size=0.2, random_state=42)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42)

# ------------------------------
# Basic DW-PW Convolution Block
# ------------------------------
def dw_pw_block(x, k=3, d=1, c=32):
    x = DepthwiseConv2D(kernel_size=k, padding='same', dilation_rate=d)(x)
    x = Conv2D(c, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# ------------------------------
# SC-PTEM: Multi-scale branches
# ------------------------------
def sc_ptem_branches(x):
    A1 = dw_pw_block(x, k=3, d=1, c=32)
    A2 = dw_pw_block(x, k=5, d=1, c=32)
    A3 = dw_pw_block(x, k=3, d=3, c=32)
    return A1, A2, A3

# ------------------------------
# SC-PTEM: Scale-weighted fusion
# ------------------------------
def sc_ptem_fusion(A1, A2, A3):

    concat = Concatenate(axis=-1)([A1, A2, A3])

    gates = Conv2D(3, kernel_size=1, padding='same')(concat)
    gates = Softmax(axis=-1)(gates)

    g1 = Lambda(lambda x: x[...,0:1])(gates)
    g2 = Lambda(lambda x: x[...,1:2])(gates)
    g3 = Lambda(lambda x: x[...,2:3])(gates)

    F = Add()([
        Multiply()([A1, g1]),
        Multiply()([A2, g2]),
        Multiply()([A3, g3])
    ])

    return F

# ------------------------------
# TAAM: Spatial Attention (SAO)
# ------------------------------
from tensorflow.keras.layers import Lambda

def spatial_attention(x):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attn = Conv2D(1, kernel_size=1, padding='same')(concat)
    attn = Activation('sigmoid')(attn)

    return attn

# ------------------------------
# TAAM: Channel Attention
# ------------------------------
def channel_attention(x, r=8):
    c = x.shape[-1]
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(c // r, activation='relu')(gap)
    fc2 = Dense(c, activation='sigmoid')(fc1)
    return Reshape((1, 1, c))(fc2)

# ------------------------------
# TAAM Module
# ------------------------------
def taam_module(x):
    Fs = Multiply()([x, spatial_attention(x)])
    Fc = Multiply()([x, channel_attention(x)])
    F_out = Add()([Fs, Fc])
    return F_out

# ------------------------------
# Classification Head
# ------------------------------
def classification_head(x, num_classes):
    x = Conv2D(64, kernel_size=1, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return output

# ------------------------------
# Full Model (Figure-aligned)
# ------------------------------
def build_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    A1, A2, A3 = sc_ptem_branches(inputs)
    F = sc_ptem_fusion(A1, A2, A3)
    F = taam_module(F)
    outputs = classification_head(F, num_classes)
    model = Model(inputs, outputs)
    return model

# Example usage
model = build_model()
model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss=losses.CategoricalCrossentropy(),
    metrics=[metrics.CategoricalAccuracy()]
)
# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train with smaller batch size
history = model.fit(
    X_train1, y_train1,
    batch_size=16,  # Adjust based on your GPU memory
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[lr_scheduler, early_stopping]
    callbacks=[lr_scheduler]
)

import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
