# MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
# import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import KernelPCA

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

# Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list, images, labels):
  # insert code that when given a list of integers, will find the labels and images
  # and put them all in numpy arrary (at the same time, as training and testing data)
  selected_images = []
  selected_labels = []

  for number in number_list:
    for ind, label in enumerate(labels):
      if (label == number):
        selected_images.append(images[ind])
        selected_labels.append(labels[ind])
        break

  images_nparray = np.array(selected_images)
  labels_nparray = np.array(selected_labels)

  return(images_nparray, labels_nparray)


def print_numbers(images, labels, data, num_labels, model, fig):
  # insert code that when given images and labels (of numpy arrays)
  # the code will plot the images and their labels in the title.
  title = data + " Data"

  if (num_labels == 20875):
    add_title = ": [2,0,8,7,5]"
  elif (num_labels == 0.123456789):
    add_title = ": [0,1,2,3,4,5,6,7,8,9]"

  if (model == "GAU"):
    another_one = " - Gaussian Naive Bayes Model"
  elif (model == "KNN"):
    another_one = " - K Nearest Neighbors Classifier Model"
  elif (model == "MLP"):
    another_one = " - MLP Classifier Model"
  else: # model == 0
    another_one = ""

  plt.figure(fig, figsize=(10, 5))
  plt.suptitle(title + add_title + another_one)
  for ind, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(1, len(images), ind + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.xticks([])
    plt.yticks([])
  plt.show()

  return()


class_numbers = [2,0,8,7,5]
# Part 1
class_number_images, class_number_labels = dataset_searcher(class_numbers, X_train, y_train)
# Part 2
print_numbers(class_number_images, class_number_labels, "Original", 20875, 0, 1)



model_1 = GaussianNB()

# however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

# Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
# Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped)


def OverallAccuracy(results, actual_values):
  # Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  correct = np.sum(results == actual_values)
  total = len(results)
  accuracy = correct / total

  return(accuracy)


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall result of the Gaussian model is " + str(Model1_Overall_Accuracy))


# Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, X_train, y_train)
allnumbers_images_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
allnumbers_results = model_1.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results, "Original", 0.123456789, "GAU", 2)


# Part 6
# Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall result of the K Nearest Neighbors model is " + str(Model2_Overall_Accuracy))

allnumbers_results_KNN = model_2.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_KNN, "Original", 0.123456789, "KNN", 3)

# Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall result of the MLP Classifier model is " + str(Model3_Overall_Accuracy))

allnumbers_results_MLP = model_3.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_MLP, "Original", 0.123456789, "MLP", 4)


# Part 8
# Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


# Part 9-11
# Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

# GaussianNB Poisoned
model_1_poison = GaussianNB()
model_1_poison.fit(X_train_poison_reshaped, y_train)
model1_results_poison = model_1_poison.predict(X_test_reshaped)
Model1_Poison_Overall_Accuracy = OverallAccuracy(model1_results_poison, y_test)
print("The overall poisoned result of the Gaussian model is " + str(Model1_Poison_Overall_Accuracy))

allnumbers_results_G_poison = model_1_poison.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_G_poison, "Poisioned", 0.123456789, "GAU", 5)

# KNN Poisoned
model_2_poison = KNeighborsClassifier(n_neighbors=10)
model_2_poison.fit(X_train_poison_reshaped, y_train)
model2_results_poison = model_2_poison.predict(X_test_reshaped)
Model2_Poison_Overall_Accuracy = OverallAccuracy(model2_results_poison, y_test)
print("The overall poisoned result of the K Nearest Neighbors model is " + str(Model2_Poison_Overall_Accuracy))

allnumbers_results_KNN_poison = model_2.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_KNN_poison, "Poisioned", 0.123456789, "KNN", 6)

# MLP Poisoned
model_3_poison = MLPClassifier(random_state=0)
model_3_poison.fit(X_train_poison_reshaped, y_train)
model3_results_poison = model_3_poison.predict(X_test_reshaped)
Model3_Poison_Overall_Accuracy = OverallAccuracy(model3_results_poison, y_test)
print("The overall poisoned result of the MLP Classifier model is " + str(Model3_Poison_Overall_Accuracy))

allnumbers_results_MLP_poison = model_3.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_MLP_poison, "Poisioned", 0.123456789, "MLP", 7)


# Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

kernel_pca = KernelPCA(kernel='rbf', gamma=0.001, alpha=5e-3, fit_inverse_transform=True)
_= kernel_pca.fit(X_train_poison_reshaped)
X_train_denoised_inverse = kernel_pca.inverse_transform(kernel_pca.transform(X_train_poison_reshaped))


# Part 14-15
# Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
# Explain how the model performances changed after the denoising process.

# GaussianNB Denoised
model_1_denoised = GaussianNB()
model_1_denoised.fit(X_train_denoised_inverse, y_train)
model1_results_denoised = model_1_denoised.predict(X_test_reshaped)
Model1_Denoised_Overall_Accuracy = OverallAccuracy(model1_results_denoised, y_test)
print("The overall denoised result of the Gaussian model is " + str(Model1_Denoised_Overall_Accuracy))

allnumbers_results_G_denoised = model_1_denoised.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_G_denoised, "Denoised", 0.123456789, "GAU", 8)

# KNN Denoised
model_2_denoised = KNeighborsClassifier(n_neighbors=10)
model_2_denoised.fit(X_train_denoised_inverse, y_train)
model2_results_denoised = model_2_denoised.predict(X_test_reshaped)
Model2_Denoised_Overall_Accuracy = OverallAccuracy(model2_results_denoised, y_test)
print("The overall denoised result of the K Nearest Neighbors model is " + str(Model2_Denoised_Overall_Accuracy))

allnumbers_results_KNN_denoised = model_2.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_KNN_denoised, "Denoised", 0.123456789, "KNN", 9)

# MLP Denoised
model_3_denoised = MLPClassifier(random_state=0)
model_3_denoised.fit(X_train_denoised_inverse, y_train)
model3_results_denoised = model_3_denoised.predict(X_test_reshaped)
Model3_Denoised_Overall_Accuracy = OverallAccuracy(model3_results_denoised, y_test)
print("The overall denoised result of the MLP Classifier model is " + str(Model3_Denoised_Overall_Accuracy))

allnumbers_results_MLP_denoised = model_3.predict(allnumbers_images_reshaped)
print_numbers(allnumbers_images, allnumbers_results_MLP_denoised, "Denoised", 0.123456789, "MLP", 10)