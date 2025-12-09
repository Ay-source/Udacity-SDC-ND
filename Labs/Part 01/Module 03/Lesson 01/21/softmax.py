import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    L = [np.exp(L) for member in L]
    total = np.sum(L)
    result = [L/total for element in L]
    return result