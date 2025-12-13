import numpy as np

# Number of records and input units
features = np.zeros((1, 3))
print(features.shape)
n_records, n_inputs = features.shape
# Number of hidden units
n_hidden = 2
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))
print(weights_input_to_hidden)