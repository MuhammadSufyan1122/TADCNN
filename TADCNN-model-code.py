
```
 x = DepthwiseConv2D(k, padding='same', dilation_rate=d)(x)
 x = Conv2D(c, 1, padding='same')(x)
 x = BatchNormalization()(x)
 return ReLU()(x)
```

````|
| **Multi-scale branches (A¹, A², A³)** | Capture texture at different scales | ```python
def sc_ptem_branches(x):
    A1 = dw_pw_block(x, k=3, d=1, c=32)
    A2 = dw_pw_block(x, k=5, d=1, c=32)
    A3 = dw_pw_block(x, k=3, d=3, c=32)
    return A1, A2, A3
``` |
| **Per-pixel gating (SC-PTEM)** | Scale-weighted sum (Eqs. 6–7) | ```python
def sc_ptem_fusion(A1, A2, A3):
    concat = Concatenate(axis=-1)([A1, A2, A3])
    g = Conv2D(3, 1, padding='same')(concat)
    g = Softmax(axis=-1)(g)
    g1, g2, g3 = tf.split(g, 3, axis=-1)
    F = Add()([
        Multiply()([A1, g1]),
        Multiply()([A2, g2]),
        Multiply()([A3, g3])
    ])
    return F
``` |
| **SAO (Max + Avg)** | Spatial attention | ```python
def spatial_attention(x):
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx  = tf.reduce_max(x, axis=-1, keepdims=True)
    s   = Concatenate()([avg, mx])
    s   = Conv2D(1, 1, padding='same')(s)
    return Activation('sigmoid')(s)
``` |
| **Channel attention (FC, r)** | Channel re-weighting | ```python
def channel_attention(x, r=8):
    c = x.shape[-1]
    g = GlobalAveragePooling2D()(x)
    g = Dense(c//r, activation='relu')(g)
    g = Dense(c, activation='sigmoid')(g)
    return Reshape((1,1,c))(g)
``` |
| **TAAM Module** | Fuse spatial & channel attention | ```python
def taam_module(x):
    Fs = Multiply()([x, spatial_attention(x)])
    Fc = Multiply()([x, channel_attention(x)])
    return Add()([Fs, Fc])
``` |
| **Classification Head** | Final prediction | ```python
def classifier_head(x, n):
    x = Conv2D(64, 1, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    return Dense(n, activation='softmax')(x)
``` |
| **Full Model (Figure-aligned)** | End-to-end network | ```python
def build_model(input_shape=(224,224,3), n_classes=5):
    inp = Input(input_shape)
    A1, A2, A3 = sc_ptem_branches(inp)
    F  = sc_ptem_fusion(A1, A2, A3)
    F  = taam_module(F)
    out = classifier_head(F, n_classes)
    return Model(inp, out)
``` |

````
