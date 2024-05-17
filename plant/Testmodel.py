import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from imutils import paths
import random
import os
import pandas as pd
import pickle
import mahotas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Load the trained model
model = load_model("Models/custom_dl_model.h5")

# Load the label encoder
le = pickle.load(open("Extras/leenc_all.pkl", 'rb'))

# Load the MinMaxScaler
scaler = pickle.load(open("Extras/scaler_all.pkl", 'rb'))


#####################################
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



def plot_probability_distribution(prediction_probs):
	plt.figure(figsize=(8, 6))
	
	if len(prediction_probs.shape) == 2 and prediction_probs.shape[0] == 1:
		prediction_probs = prediction_probs.flatten()

	num_classes = len(le.classes_)
	plt.bar(range(num_classes), prediction_probs)
	plt.xticks(range(num_classes), range(0, num_classes), rotation=45)
	plt.xlabel('Encoded Class')
	plt.ylabel('Probability')
	plt.title('Prediction Probability Distribution')
	plt.tight_layout()
	plt.savefig('Prediction_Probablity.png')


def save_images(color_img, hsv_img, segmented_img):

	cv2.imwrite("color_image.jpg", cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
	cv2.imwrite("hsv_image.jpg", cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV))
	cv2.imwrite("segmented_image.jpg", segmented_img)


def test_image(image_path, save_results=True):
	# Read and preprocess the input image
	img = cv2.imread(image_path)
	img = cv2.resize(img, (500, 500))
	bgrim = Convert_to_bgr(img)
	hsvim = Convert_to_hsv(bgrim)
	seg_image = img_segmentation(bgrim, hsvim)
	f_shape = get_shape_feats(seg_image)
	f_text = get_texture_feats(seg_image)
	f_color = get_color_feats(seg_image)
	f_combined = np.hstack([f_color, f_text, f_shape])
	input_data = scaler.transform([f_combined])

	if save_results:
		save_images(bgrim, hsvim, seg_image)

	# Make prediction
	prediction_probs = model.predict(input_data)
	print("**************")
	print(prediction_probs)
	predicted_label_idx = np.argmax(prediction_probs)
	print("^^^^^^^^^^^^^")
	print(predicted_label_idx)
	predicted_label = le.classes_[predicted_label_idx]
	print(predicted_label)

	# Plot probability distribution
	plot_probability_distribution(prediction_probs)




	return predicted_label





if __name__=="__main__":
	from tkinter.filedialog import askopenfilename

	image_path = askopenfilename()
	predicted_label, prediction_probs = test_image(image_path)

	print("Predicted Label:", predicted_label)
	print("Prediction Probabilities:", prediction_probs)

