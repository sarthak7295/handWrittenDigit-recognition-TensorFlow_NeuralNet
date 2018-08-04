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


def show_image(n, images):
    row_no = n
    a = np.array(images[row_no, :], dtype=int)
    b = np.reshape(a, (-1, 28))
    img = Image.fromarray(b)
    img.show(title=row_no)


X, Y = read_dataset()
# show_n_random_images(3, X)


# defining the imp parameters
learning_rate = 0.2
training_epocs = 2000            # no of iterations
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_classes = 10       #no of classes Mine and Rock so 2
model_path = "D:\\PycharmProjects\\TensorFlow_Models\\MNIST_Perceptron"

# defining the hidden layer and ip and op layer
n_hidden_1 = 120
n_hidden_2 = 120
n_hidden_3 = 120
n_hidden_4 = 120

# Defining my placeholders and variables : input ,weights,biases and output
x = tf.placeholder(tf.float32, [None, n_dim])
y_ = tf.placeholder(tf.float32, [None, n_classes])

# defining my model
def multilayer_perceptron(x, weights,biases):
    #hidden layer with sigmoid activation

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)

    # hidden layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)

    # hidden layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)

    # hidden layer 4
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #output layer
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer


# defining the weights and biases
# assigns random truncated values to weights and biases
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes])),
}

# it is take every neuron has a different bias for it ,  i thought one layer had only on bias, well it
# is all about your personal preference
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes])),
}

y = multilayer_perceptron(x,weights,biases)
# logits is output given by hypothesis
# labels are the actual output we know
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess,model_path)

for ctr in range(3):
    row_no = random.randint(1, 800)
    d = multilayer_perceptron(x, weights, biases)
    prediction = tf.argmax(y, 1)
    prediction_run = sess.run(prediction, feed_dict={x: X[row_no].reshape(1, 784)})
    # show_image(row_no,X)
    print('row no : ',row_no ,'predicted value ',prediction_run,'actual value',np.argmax(Y[row_no]))