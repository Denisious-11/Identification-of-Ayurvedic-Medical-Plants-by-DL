
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
import tensorflow as tf

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




# feature-descriptor-1: Hu Moments
def get_shape_feats(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def get_texture_feats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
# feature-descriptor-3: Color Histogram
def get_color_feats(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#####################################

# data = []
# labels = []
# print("[INFO] loading images...")
# img_dir=sorted(list(paths.list_images("Medicinal Leaf Dataset/Segmented Medicinal Leaf Images")))
# random.shuffle(img_dir)
# tot=len(img_dir)
# print("total-->",tot)
# print("[INFO]  Preprocessing...")
# cnt=1
# for i in img_dir:
#         image = cv2.imread(i)
#         image = cv2.resize(image, (500, 500))

#         bgrim       = Convert_to_bgr(image)
      
#         hsvim       = Convert_to_hsv(bgrim)
  
#         seg_image   = img_segmentation(bgrim,hsvim)
        


#         f_shape = get_shape_feats(seg_image)
#         f_text   = get_texture_feats(seg_image)
#         f_color  = get_color_feats(seg_image)

#         # Concatenate 

#         f_combined = np.hstack([f_color, f_text, f_shape])


#         lab=i.split(os.path.sep)[-2]
#         labels.append(lab)
#         data.append(f_combined)
#         print("image processed-->",str(cnt),"/",str(tot))
#         cnt+=1
        
# print(len(data))
# print(len(labels))
# pickle.dump(data,open("Extras/data_all.pkl",'wb'))
# pickle.dump(labels,open("Extras/labels_all.pkl",'wb'))

########################################################

data=pickle.load(open("Extras/data_all.pkl",'rb'))
data=np.array(data)

scaler = MinMaxScaler()
data=scaler.fit_transform(data)
pickle.dump(scaler,open("Extras/scaler_all.pkl",'wb'))
print("*********")
print(data.shape)
labels=pickle.load(open("Extras/labels_all.pkl",'rb'))
le=LabelEncoder()
labels=le.fit_transform(labels)
pickle.dump(le,open("Extras/leenc_all.pkl",'wb'))
print(set(labels))
print(len(set(labels)))


# print("[INFO] Splitting Datas...")
trainX, testX, trainY, testY= train_test_split(data,labels, test_size=0.25, random_state=42)

print("\nTraining Set")
print(trainX.shape)
print(trainY.shape)
print("\nTesting Set")
print(testX.shape)
print(testY.shape)


from tensorflow.keras.callbacks import LearningRateScheduler

# Define a learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch > 0:
        return lr * 0.9  # Adjust the factor as needed
    return lr

# Use the scheduler during model training
lr_schedule = LearningRateScheduler(lr_scheduler)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Build the customized deep learning model
# model = Sequential()
# model.add(Dense(512, input_shape=(trainX.shape[1],), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(set(labels)), activation='softmax'))  # Output layer with the number of classes

model = Sequential()
model.add(Dense(512, input_shape=(trainX.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # Added layer
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
epochs = 500
batch_size = 32
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[lr_schedule])

# Evaluate the model on the test set
predictions = model.predict(testX)
predictions = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(testY, predictions)
classification_rep = classification_report(testY, predictions)
conf_matrix = confusion_matrix(testY, predictions)

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_rep)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Save the trained model
model.save("Models/custom_dl_model.h5")

###########################################
from lime import lime_tabular

# Load the trained model
model = tf.keras.models.load_model("Models/custom_dl_model.h5")


# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(trainX, feature_names=[f'f{i}' for i in range(trainX.shape[1])])

# Select a random sample from the test set
sample_idx = random.randint(0, len(testX))
sample = testX[sample_idx]

# Explain the prediction using LIME
exp = explainer.explain_instance(sample, model.predict, num_features=trainX.shape[1])

feature_names = [f'f{i}' for i in range(trainX.shape[1])]
feature_weights = exp.as_list()

# Sort features based on importance
sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

# Display the top N features as a bar graph with rotated y-axis labels
top_n = 10  # Set the number of top features to display
top_features = sorted_features[:top_n]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(range(len(top_features)), [weight for _, weight in top_features], align="center")
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([feature for feature, _ in top_features], rotation=45, ha='right')  # Rotate y-axis labels
ax.invert_yaxis()  # Invert the y-axis for better visualization
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} LIME Feature Importance')

# Add values on top of the bars for better visibility
for bar in bars:
    yval = bar.get_y() + bar.get_height() / 2
    plt.text(bar.get_width(), yval, round(bar.get_width(), 4), va='center', ha='left', fontsize=8)

plt.show()