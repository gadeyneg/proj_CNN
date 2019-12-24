import numpy as np
import math

def sum_square(y_pred, y_target):
    if type(y_pred) != np.ndarray or type(y_target) != np.ndarray:
        print("Type problem : target should be an array")
        return
    if y_pred.shape != y_target.shape:
        print("Shape problem : target should be of shape ", y_pred.shape, " but ", y_target.shape, "has been given." )
        return()
    loss = 1/2*sum([(a-b)**2 for a, b in zip(y_pred.flatten(), y_target.flatten())])
    loss_der = (y_target - y_pred)
    return loss, loss_der 

def softmax(y):
    exp_y = list(map(math.exp, y))
    return np.asarray([ val / sum(exp_y) for val in exp_y])

def categorical_crossentropy(y_pred, y_target):
    y_pred_sft = softmax(y_pred)
    loss = - sum([ti * math.log(pi) for pi, ti in zip(y_pred_sft.flatten(), y_target.flatten())])
    loss_der = y_target - y_pred_sft
    return loss, loss_der
