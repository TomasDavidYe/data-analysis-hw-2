####The project contrains three file with code

#####1) helper_methods.py
this file constains methods for preprocessing of the data

#####2) custom_ml_library.py
Here there is a class representing the neural network. When calling the constructor,
you pass in the structure of the neural network and it creates an abstract computational diagram.
When running some of the class methods, data flows through the diagram as expected.

#####3) program.py
Running this file will start a full lifecycle of the model. We get the processed development data, 
split the data randomly, train the model and test the accuracy. This process is repeated arbitrary 
and then average errors are calculated. Finaly, we make the model predict prices for the unknown datapoints. 