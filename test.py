import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import random


def read_dataset():
    # we now save the first  col in y since the rest are features ie pixels in x
    dataset = pd.read_csv("DataSet/train.csv")
    X = dataset[dataset.columns[1:]].values
    y = dataset[dataset.columns[0]].values
    # print(y[0:5])
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return X, Y


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded_matrix = np.zeros((n_labels, n_unique_labels))
    one_hot_encoded_matrix[np.arange(n_labels), labels] = 1
    return one_hot_encoded_matrix


def show_n_random_images(n, images):
    for counter in range(n):
        row_no = random.randint(1,500)
        a = np.array(images[row_no, :], dtype=int)
        b = np.reshape(a, (-1, 28))
        img = Image.fromarray(b)
        img.show()


X, Y = read_dataset()
show_n_random_images(3, X)


