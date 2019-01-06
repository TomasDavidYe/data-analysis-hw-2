from helper_methods import *
from custom_ml_library import NeuralNet


[development_data, evaluation_data] = get_data()
development_data = transformation_of_dev_data(development_data)
evaluation_data = transformation_of_eval_data(evaluation_data)

num_of_iter = 1
learning_rate = 0.01

my_net = NeuralNet(learning_rate)

for i in range(0, num_of_iter):
    [trainX, trainY, testX, testY] = split_development_data(development_data)












