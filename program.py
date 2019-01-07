from helper_methods import *
from custom_ml_library import NeuralNet


[development_data, evaluation_data] = get_data()
mean = development_data.mean()
std = development_data.std()
development_data = transformation_of_dev_data(development_data)
evaluation_data = transformation_of_eval_data(evaluation_data)

# num_of_iter = 120
# learning_rate = 0.00001
input_layer_dim = len(development_data.columns) - 1
hidden_layer_dims = [5, 5]
output_layer_dim = 1

trainingRMEs = []
trainingMAEs = []

testingRMSEs = []
testingMAEs = []


def append_errors(trainingRMSE, trainingMAE, testingRMSE, testingMAE):
    trainingRMEs.append(trainingRMSE)
    trainingMAEs.append(trainingMAE)
    testingRMSEs.append(testingRMSE)
    testingMAEs.append(testingMAE)

def run_single_session(num_of_iter, learning_rate, index = 1):
    print("Running session number ", index)
    net = NeuralNet(learning_rate=learning_rate, input_layer_dim=input_layer_dim,
                       hidden_layer_dims=hidden_layer_dims, output_layer_dim=output_layer_dim)

    [train_X, train_Y, test_X, test_Y] = split_development_data(development_data)
    net.train_model(train_X=train_X, train_Y=train_Y, num_iter=num_of_iter)

    [trainingRMSE, trainingMAE] = net.get_errors(X=train_X, y=train_Y)
    [testingRMSE, testingMAE] = net.get_errors(X=test_X, y=test_Y)
    append_errors(trainingRMSE, trainingMAE, testingRMSE, testingMAE)

    print("train RMSE = ", trainingRMSE)
    print("test RMSE  = ", testingRMSE)
    print("train MAE  = ", trainingMAE)
    print("test MAE   =", testingMAE)

    print("\n\n")
    return


for i in range(1,11):
    run_single_session(num_of_iter=1000, learning_rate=0.00000001, index=i)
