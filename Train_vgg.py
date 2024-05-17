
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import MinMaxScaler
import mahotas
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Converting each image to RGB from BGR format
bins= 8
def Convert_to_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img


# Conversion to HSV image format from RGB

def Convert_to_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



# image segmentation

# for extraction of green and brown color


def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result



####################################

data = []
labels = []
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("Medicinal Leaf Dataset/Segmented Medicinal Leaf Images")))
random.shuffle(img_dir)
tot=len(img_dir)
print("total-->",tot)
print("[INFO]  Preprocessing...")
cnt=1
for i in img_dir:
        image = cv2.imread(i)
        image = cv2.resize(image, (224, 224))

        bgrim       = Convert_to_bgr(image)
      
        hsvim       = Convert_to_hsv(bgrim)
  
        seg_image   = img_segmentation(bgrim,hsvim)


        lab=i.split(os.path.sep)[-2]
        labels.append(lab)
        data.append(seg_image)
        print("image processed-->",str(cnt),"/",str(tot))
        cnt+=1
        
print(len(data))
print(len(labels))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Build a sequential model on top of VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

# Save the model
model.save("Models/vgg16_model.h5")

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Generate and print classification report
class_report = classification_report(y_test, y_pred_classes, target_names=le.classes_)
print("Classification Report:\n", class_report)
