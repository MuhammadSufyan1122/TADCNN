
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

# Set seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_random_seeds(42)


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
    g1, g2, g3 = tf.split(gates, num_or_size_splits=3, axis=-1)
    F = Add()([
        Multiply()([A1, g1]),
        Multiply()([A2, g2]),
        Multiply()([A3, g3])
    ])
    return F

# ------------------------------
# TAAM: Spatial Attention (SAO)
# ------------------------------
def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
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



