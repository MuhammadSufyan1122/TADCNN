# TADCNN for Lung cancer classification 

# introduction

Chest CT scan images hold invaluable diagnostic potential in identifying various pulmonary conditions, including malignant tumors. Our project aims to streamline the classification process of these images into four distinct classes: 'adenocarcinoma', 'large cell carcinoma', 'normal', and 'squamous cell carcinoma'. In this article we have presented texture aware deep convolutional network for lung cell classification.

## Dataset

Dataset for this Project is taken from Kaggle. Here is the Dataset [Link](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data).

## Dataset Information

* Images are not in dcm format, the images are in jpg or png to fit the model.
* Data contain 3 chest cancer types which are Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma , and 1 folder for the normal cell.
* Data folder is the main folder that contain all the step folders inside Data folder are test , train , valid.

* test represent testing set
* train represent training set
* valid represent validation set
* training set is 72%
* testing set is 18%
* validation set is 10%
#### Dataset Details<a id='dataset-details'></a>
<pre>
Dataset Name            : Chest CT-Scan images Dataset (Adenocarcinoma vs Large cell carcinoma vs Squamous cell carcinoma vs Normal)
Number of Class         : 4
Number/Size of Images   : Total      : 1000 (124 MB)
                          Training   : 720
                          Testing    : 180
                          Validation : 100 
</pre>

## Installation
The Code is written in Python 3.9.19. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.
## Run Locally
```bash
conda create -p env python=3.9 -y
```
```bash
conda activate ./env
```
### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```
### Step 4 - Load dataset and Preprocessing
 

```bash
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
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
```



# Workflow of Methdology
<br>

All images are resized to 224×224 pixels to standardize input shape and reduce computational cost (bilinear
interpolation for RGB, area interpolation for CT slices). For LC25000, RGB values are converted to float32 and
scaled to [0, 1] via division by 255. For IQ-OTH/NCCD, images are likewise scaled to [0, 1] for consistency. After
preprocessing, the data are partitioned at the patient level into 72% training, 18% validation, and 10% test (i.e., 90%
train/val split with an 80:20 split), using stratified sampling and a fixed random seed for reproducibility. During training
we apply light augmentation (random horizontal/vertical flips and small rotations) to reduce overfitting. Fig. 3 presents
a schematic overview of the proposed methodology.

<img src="Images/Workflow.png" border="0">
</br>


# Proposed Model
<br>

Overview of the proposed TADCNN. Left — SC-PTEM: three depthwise-separable branches with different
receptive fields extract multiscale features. Their outputs are fed to a small gating path (concat → 1×1 conv → gating
head → softmax) that produces per-pixel weights; the branches are reweighted and merged into a single fused feature
map. Middle — TAAM: two attentions run in parallel—spatial attention (avg/max over channels → 1×1 conv → sigmoid)
and channel attention (global average pooling → small MLP → sigmoid); their outputs are added to refine the features
without changing tensor size. Right — Classification head: an optional 1×1 projection, global average pooling, and a linear
layer with softmax produce class probabilities.


<img src="Images/TADCNN-model.png" border="0">
</br>

# Results
## On LC25000 Dataset
<br>

Comparison of TADCNN model with other deep learning models based on convergence in the accuracy while
training and testing the LC25000 dataset; (a) Train accuracy curves, (b) Test accuracy curves, (c) Train loss curves,(d)
Test loss curves

<img src="Images/acc LC25000.png" border="0">
</br>

<br>

Comparison of TADCNN model with other deep learning models based on the confusion matrices for the LC25000
dataset; (a) Proposed TADCNN, (b) DenseNet121, (c) Efficient- NetV2L ,(d) Xception, (e)InceptionResNetV2, (f) VGG19,
(g) MobileNetV2, (h) ShuffleNetV2

<img src="Images/CM LC25000.png" border="0">
</br>

<br>
Classification results showing actual vs predicted tissue types, with high confidence levels across different
conditions: Colon-aca, Colon-n, Lung-aca, Lung-n, and Lung-scc

<img src="Images/prediction LC25000.png" border="0">
</br>
<br>
Examples of predictions with varying confidence scores on challenging LC25000 patches.

<img src="Images/low performance LC25000.png" border="0">
</br>

## On IQ-OTH/NCCD Dataset

<br>
Comparison of TADCNN model with other deep learning models based on convergence in the accuracy while
training and testing the IQ-OTH/NCDD dataset; (a) Train accuracy curves, (b) Test accuracy curves, (c) Train loss
curves,(d) Test loss curves.

<img src="Images/accuracy curves IQ-OTHNCDD.png" border="0">
</br>

<br>
Comparison of the TADCNN model with other deep learning models based on the confusion matrices
for the IQ-OTH/NCDD dataset; (a) Proposed TADCNN, (b) DenseNet121, (c) Efficient- NetV2L ,(d) Xception,
(e)InceptionResNetV2 , (f) VGG19, (g) MobileNetV2, (h)ShuffleNetV2.

<img src="Images/CM IQ-OTHNCDD.png" border="0">
</br>

<br>
Classification results showing actual vs predicted tissue types, with high confidence levels across different
conditions: benign, Malignant, Normal.

<img src="Images/prediction on IQ-N data.png" border="0">
</br>
