
import keras
from keras.models import load_model
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CATEGORIES = ["Dog", "Cat"]

image = r"H:\Projects\Cats and Dogs-Classification\test1\128.jpg"

def prepare(filepath):

    image_size = 120

    image_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image_array, (image_size, image_size))

    return new_array.reshape(-1, image_size, image_size, 1)

model = keras.models.load_model(r"H:\Projects\Cats and Dogs-Classification\cnn_classification.model")


prediction = model.predict([prepare(image)])
print(int(prediction))