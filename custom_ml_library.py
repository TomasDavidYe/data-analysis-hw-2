import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNet:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train_model(self, X, y):
        return

    def predict(self, X):
        return None



















a = tf.constant(3.0)
b = tf.constant(4.0)

c = a*b
d = tf.constant(5.0)
e = c + d

u = tf.constant(9.0)
v = tf.constant(13.0)

x = e*u + v

sess = tf.Session()
file_writer = tf.summary.FileWriter('./graph', sess.graph)

result = sess.run(c)
sess.close()
