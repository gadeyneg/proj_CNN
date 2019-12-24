#%%

import numpy as np
import Layers
import Losses
import math
from importlib import reload
reload(Layers)

class NeuralNetwork:
    def __init__(self, loss_func = Losses.sum_square):
        self.input_shape = 0
        self.output_shape = 0
        self.layers = []
        self.loss = lambda x,y : loss_func(x,y)[0]
        self.loss_derivative = lambda x,y : loss_func(x,y)[1]
        self.added_first_layer = False        


    def add_layer(self, layer):
        if not self.added_first_layer:
            self.input_shape = layer.get_shape()
            self.added_first_layer = True            
        self.layers.append(layer)
        self.output_shape = layer.get_shape()

    def predict(self, X):
        self.layers[0].compute(X)
        for i in range(1, len(self.layers)):
            self.layers[i].compute()
        return(self.layers[-1].get_output())

    def backprop(self, prediction, target):
        D = self.loss_derivative(prediction, target)
        for i in range(len(self.layers)-1, 0, -1):
            D = self.layers[i].compute_backprop(D)

    def run(self, X, Y):
        for x,y in zip(X, Y):
            y_pred = self.predict(x)
            self.backprop(y_pred, y)


# %%

import Layers
import random

model = NeuralNetwork(loss_func=Losses.sum_square)

l1 = Layers.Input_2D((8,8,1))
model.add_layer(l1)

l2 = Layers.Conv2D(l1, 5)
model.add_layer(l2)

#%%
if __name__ == "__main__" :

    import Layers
    import random

    model = NeuralNetwork(loss_func=Losses.sum_square)

    l1 = Layers.Input_2D((8,8,1))
    model.add_layer(l1)

    l2 = Layers.Conv2D(l1, 5)
    model.add_layer(l2)

    a = [0,0,1]
    b = [0,2,0]
    c = [3,0,0]
    d = [0,0,0]

    a = 1
    b = 5
    c = 3
    d = 2

    X = np.random.rand(8,8)

    X = X.reshape(X.shape + (1,))

    print(model.predict(X))
# %%


