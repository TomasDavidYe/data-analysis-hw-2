import os
import tensorflow as tf
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNet:

    def __init__(self, learning_rate, input_layer_dim, hidden_layer_dims, output_layer_dim):
        self.input_layer_dim = input_layer_dim
        self.output_layer_dim = output_layer_dim
        self.learning_rate = learning_rate
        self.depth = len(hidden_layer_dims)
        self.weights = self.create_weight_matrices(input_layer_dim=input_layer_dim, hidden_layer_dims=hidden_layer_dims, output_layer_dim=output_layer_dim)
        self.biases = self.create_bias_vectors(hidden_layer_dims=hidden_layer_dims, output_layer_dim=output_layer_dim)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_weight_matrices(self, input_layer_dim, hidden_layer_dims, output_layer_dim):
        matrices = []
        matrices.append(self.get_random_matrix(shape=[input_layer_dim, hidden_layer_dims[0]], name="WEIGHTS_1"))

        for i in range(0, self.depth - 1):
            row_count = hidden_layer_dims[i]
            column_count = hidden_layer_dims[i + 1]
            shape = [row_count, column_count]
            matrices.append(self.get_random_matrix(shape=shape, name="WEIGHTS_{}".format(i + 2)))

        matrices.append(self.get_random_matrix(shape=[hidden_layer_dims[self.depth - 1], output_layer_dim], name="WEIGHTS_{}".format(self.depth + 1)))

        return matrices

    def create_bias_vectors(self, hidden_layer_dims, output_layer_dim):
        vectors = []
        for i in range(0, self.depth):
            shape = [1, hidden_layer_dims[i]]
            vectors.append(self.get_random_matrix(shape=shape, name="BIAS_{}".format(i + 1)))
        vectors.append(self.get_random_matrix(shape=[1, output_layer_dim], name="BIAS_{}".format(self.depth + 1)))

        return vectors


    def get_random_matrix(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, name="RAND_{}x{}".format(shape[0], shape[1])), name=name)

    def apply_net(self, X):
        temp = X
        for i in range(self.depth):
            weight_matrix = self.weights[i]
            bias_vector = self.biases[i]
            temp = tf.matmul(temp, weight_matrix, name="MULTIPLY_{}".format(i + 1))  # Here we multiplying by weight matrix
            temp = tf.add(temp, bias_vector, name="ADD_{}".format(i + 1))  # Here we add the bias vector
            temp = tf.nn.relu(temp, name="ReLU_{}".format(i + 1))

        final_weight_matrix = self.weights[self.depth]
        final_bias_matrix = self.biases[self.depth]
        temp = tf.matmul(temp, final_weight_matrix, name="MULTIPLY_{}".format(self.depth + 1))
        return tf.add(temp, final_bias_matrix, name="OUTPUT")

    def train_model(self, train_X, train_Y, num_iter):
        print("Training model")
        [_X, _y] = self.get_input_placeholders()
        _computed_y = self.apply_net(_X)
        _cost = tf.reduce_mean(tf.square(_y - _computed_y))
        _train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(_cost)

        for i in range(num_iter):
            self.session.run([_cost, _train], {_X: train_X, _y: train_Y})
        return

    def cost(self, X, y):
        [_X, _y] = self.get_input_placeholders()
        _computed_y = self.apply_net(_X)
        _cost = tf.reduce_mean(tf.square(_y - _computed_y))
        return self.session.run(_cost, {_X: X, _y: y})

    def calculate_RMSE(self, X, y):
        return np.sqrt(self.cost(X,y))


    def calculate_MAE(self, X, y ):
        [_X, _y] = self.get_input_placeholders()
        _computed_y = self.apply_net(_X)
        _MAE = tf.reduce_mean(tf.abs(_y - _computed_y))
        return self.session.run(_MAE, {_X: X, _y: y})

    def get_errors(self, X, y):
        return [self.calculate_RMSE(X, y), self.calculate_MAE(X, y)]

    def predict(self, X):
        [_X, _y] = self.get_input_placeholders()
        return self.session.run(self.apply_net(_X), {_X: X})

    def get_input_placeholders(self):
        return [tf.placeholder(tf.float32, shape=[None, self.input_layer_dim]), tf.placeholder(tf.float32, shape=[None, self.output_layer_dim]) ]


    def close_session(self):
        self.session.close()
        return





# input_layer_dim = 7
# hidden_layer_dims = [5, 5]
# output_layer_dim = 1
# learning_rate = 0.1
#
# X = tf.placeholder(tf.float32, [None, input_layer_dim], name="INPUT")
# session = tf.Session()
# test_net = NeuralNet(learning_rate, input_layer_dim, hidden_layer_dims, output_layer_dim)
# y = test_net.apply_net(X)
# FileWriter = tf.summary.FileWriter('./graph', session.graph)
#
# init = tf.global_variables_initializer()
# session.run(init)
# session.run(y, {X: pd.DataFrame(data=[0.0 for i in range(0, 7)]).transpose()})
# session.close()
#


