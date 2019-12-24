#%%
import numpy as np
import math
import sys

def reLu(x):
    if x < 0:
        return 0, 0
    return x, 1

def sigm(x):
    try:
        s = 1/(1 + math.exp(-x))
    except OverflowError:
        s = float('inf')
    return s, s * (1-s)

def identity(x):
    return x, 1

class Dense_1D:
    def __init__(self, previous_layer, size, activation = identity):
        self.size = size
        self.weights = np.random.rand(size, len(previous_layer.output.flatten())).astype(np.float64)/size
        self.output = np.zeros((size)).astype(np.float64)
        self.previous_layer = previous_layer
        self.activation = lambda x: activation(x)[0]
        self.activation_derivate = lambda x: activation(x)[1]
        self.bias = np.random.rand(size).astype(np.float64)

    def compute(self):
        for i in range(self.size):
            self.output[i] = self.activation(sum( [self.previous_layer.output.flatten()[j] * self.weights[i, j] for j in range(len(self.previous_layer.output.flatten()))] ) + self.bias[i])
    
    def get_output(self):
        return self.output

    def get_shape(self):
        return self.size

    def compute_backprop(self, D):
        deltas = []
        for i in range(self.size):
            Delta = D[i] * self.activation_derivate(self.output[i])
            deltas.append(Delta)
            Delta_W = Delta * self.previous_layer.output.flatten()
            self.bias[i] += Delta
            self.weights[i, :] += Delta_W
        return np.dot(np.asarray(deltas), self.weights)


class Input_1D:
    def __init__(self, size):
        self.size = size
        self.output = np.zeros((size))
    
    def compute(self, vect):
        if type(vect) != np.ndarray:
            sys.exit("Type problem : input of model should be numpy array, not " + str(type(vect)))
        if len(vect) != self.size:
            sys.exit("Shape problem : input size is " + str(self.size) + " and model was given " + str(len(vect)))
        self.output = vect
        # print(self.output)

    def get_output(self):
        return self.output

    def get_shape(self):
        return self.size

class Input_2D:
    def __init__(self, shape):
        if len(shape) != 3:
            sys.exit('Wrong usage : specify number of channels')
        self.shape = shape
        self.output = np.zeros((shape))
    
    def compute(self, arr):
        if type(arr) != np.ndarray:
            sys.exit("Type problem : input of model should be numpy array, not " + str(type(arr)))
        if arr.shape != self.shape:
            sys.exit("Shape problem : input size is " + str(self.shape) + " and model was given " + str(arr.shape))
        self.output = arr

    def get_output(self):
        return self.output

    def get_shape(self):
        return self.shape 

class MaxPool2D:
    def __init__(self, previous_layer):
        h, w, f = previous_layer.get_shape()
        self.shape = ( h//2, w//2, f)
        self.output = np.zeros((self.shape)).astype(np.float64)
        self.previous_layer = previous_layer

    def compute(self):
        h,w, f = self.previous_layer.get_shape()
        for i in range(0,h-1, 2):
            for j in range(0, w-1, 2):
                self.output[i//2, j//2] = [self.previous_layer.output[i:i+2, j:j+2, f_i].max() for f_i in range(f)]

    def get_output(self):
        return self.output

    def get_shape(self):
        return self.shape

    def compute_backprop(self, D):
        return D

class Conv2D:
    def __init__(self, previous_layer, filter_number, padding = "valid"):
        h, w, channels = previous_layer.get_shape()
        self.filters = np.random.rand(filter_number, 3, 3, channels) / 9
        # I assume that there is multiple channels, therefore len is 3
        if padding == "valid":
            self.output = np.zeros(( h-2, w-2, filter_number))
        else:
            # only valid and same padding here
            self.output = np.zeros((h, w, filter_number))
        self.shape = self.output.shape
        self.previous_layer = previous_layer
        self.padding = padding

    def compute(self):
        if self.padding == "same":
            #do something
            sys.exit("same padding not implemented yet")
        h, w, filters = self.output.shape
        for f in range(filters):
            filt = self.filters[f]
            for i in range(h):
                for j in range(w):
                    self.output[i,j,f] = np.sum(filt * self.previous_layer.output[i:i+3, j:j+3])
        
    def get_output(self):
        return self.output

    def get_shape(self):
        return self.shape

    def compute_backprop(self, D):
        return D


# %%
