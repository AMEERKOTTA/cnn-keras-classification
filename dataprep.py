import numpy as np
import matplotlib.pyplot as plt

import cv2
import os


Data_Dir = r"H:\Projects\Cats and Dogs-Classification\train/"

# Importing Data Directory and Reading Data #
CATEGORIES = ["Dogs", "Cats"]

for input in CATEGORIES:

    path = os.path.join(Data_Dir, input)
    for image in os.listdir(path):

        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

        plt.imshow(image_array, cmap = "gray")
        plt.show()

        break
    break

# Resizing the Image #
image_size = 120

new_array = cv2.resize(image_array, (image_size,image_size))

plt.imshow(new_array, cmap = "gray")
plt.show()

# Preperation of Training Data #

training_data = []

def create_training_data():

    for input in CATEGORIES:

        path = os.path.join(Data_Dir, input)
        class_num = CATEGORIES.index(input)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(image_array, (image_size,image_size))
                training_data.append([new_array, class_num])

            except Exception as e:

                pass

create_training_data()
print(len(training_data))

# Shuffle the Data #

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)


# Assigning Features and Labels #

X = [] #features
y = [] #labels

for features, labels in training_data:
    X.append(features)
    y.append(labels)

print(X[0].reshape(-1, image_size, image_size, 1))

X = np.array(X).reshape(-1, image_size, image_size, 1)

# Saving the Preprocessed Files into the Directory #

import pickle

pickle_out = open("H:\Projects\Cats and Dogs-Classification\X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("H:\Projects\Cats and Dogs-Classification\y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()