from helper_methods import *
import tensorflow

[development_data, evaluation_data] = get_data()
development_data = transformation_of_features(development_data)
