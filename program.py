from helper_methods import *
from custom_ml_library import NeuralNet


[development_data, evaluation_data] = get_data()
mean = development_data.mean()
std = development_data.std()
development_data = transformation_of_dev_data(development_data)
evaluation_data = transformation_of_eval_data(evaluation_data)

num_of_iter=3000
learning_rate=0.00000001
input_layer_dim = len(development_data.columns) - 1
hidden_layer_dims = [6, 6]
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


def make_predictions(X, name):
    net = NeuralNet(learning_rate=learning_rate, input_layer_dim=input_layer_dim,
                    hidden_layer_dims=hidden_layer_dims, output_layer_dim=output_layer_dim)
    [train_X, train_Y, test_X, test_Y] = split_development_data(development_data)
    net.train_model(train_X=train_X, train_Y=train_Y, num_iter=num_of_iter)
    index = X.index
    predictions = net.predict(X=X)
    net.close_session()

    filename = name + '.csv'
    pd.DataFrame(index=index,data=predictions).to_csv('./output/' + filename)

def run_single_session(num_of_iter, learning_rate, index = 1):
    print("\n\nRunning session number ", index)
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

    net.close_session()
    return


def run_multiple_sessions(num_sessions):
    for i in range(1, num_sessions + 1):
        run_single_session(num_of_iter=num_of_iter, learning_rate=learning_rate, index=i)

    print("\n\n")
    print("avarage trainRMSE = ", pd.Series(trainingRMEs).mean())
    print("avarage testRMSE  = ", pd.Series(testingRMSEs).mean())
    print("avarage trainMAE  = ", pd.Series(trainingMAEs).mean())
    print("avarage testMAE   =", pd.Series(testingMAEs).mean())
    print("\n\n")

run_multiple_sessions(10)
make_predictions(X = evaluation_data, name='predictions')