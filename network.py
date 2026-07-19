import numpy as np
from layer import Layer
import math

class Network:
    def __init__(self, num_neurons_per_layer, num_layers, learning_rate, n_inputs, n_outputs, activation_function='relu'):
        if activation_function.lower().replace(' ', '') not in ['relu', 'leakyrelu', 'linear', 'sigmoid', 'tanh', 'softplus', 'elu']:
            raise ValueError('Activation Function Not Recognised, maybe try:\nRelu, Leaky Relu, Linear, Sigmoid, Tanh, SoftPlus or ELU')
        
        self.activation_function = activation_function.lower().replace(' ', '')

        self.neuron_numbers = num_neurons_per_layer

        # Number of inputs, Number of outputs
        self.layers = [Layer(n_inputs, num_neurons_per_layer, self.activation_function)]
        self.layers.extend([Layer(num_neurons_per_layer, num_neurons_per_layer, self.activation_function) for _ in range(num_layers)])
        self.layers.append(Layer(num_neurons_per_layer, n_outputs, self.activation_function))

        self.learning_rate = learning_rate

    def activation(self, x):
        # Chooses the correct activation function
        match self.activation_function:
            case 'relu':
                return np.maximum(0, x)
            
            case 'leakyrelu':
                return np.where(x > 0, x, x * 0.01)
                
            case 'linear':
                return x
            
            case 'sigmoid':
                return 1/(1 + np.exp(-x))
            
            case 'tanh':
                return (2/(1 + np.exp(-2*x))) - 1
            
            case 'softplus':
                return np.log(1 + np.exp(x))
            
            case 'elu':
                return np.where(x > 0, x, np.exp(x) - 1)


    # For the back propogation
    def activation_grad(self, y):
        # Chooses the correct activation function
        match self.activation_function:
            case 'relu':
                return np.where(y > 0, 1.0, 0.0)
            
            case 'leakyrelu':
                return np.where(y > 0, 1.0, 0.01)
                
            case 'linear':
                return np.ones_like(y)
            
            case 'sigmoid':
                sig = 1.0 / (1.0 + np.exp(-np.clip(y, -500, 500)))
                return sig * (1.0 - sig)
            
            case 'tanh':
                t = (2/(1 + np.exp(-2*y))) - 1
                return 1.0 - t ** 2
            
            case 'softplus':
                return 1.0 / (1.0 + np.exp(-np.clip(y, -500, 500)))
            
            case 'elu':
                return np.where(y > 0, 1.0, np.exp(np.clip(y, -500, 500)))
    
    
    def forward(self, x):
        inputs = x
        for index, layer in enumerate(self.layers):
            inputs = layer.forward_pass(inputs)
            if index != len(self.layers) - 1:
                inputs = self.activation(inputs)

        return inputs
    
    def backward(self, y_pred, y_true):
        loss_grad = (y_pred - y_true)
        loss_grad *= 2/y_true.size
        reversed_layers = list(reversed(self.layers))
        gradients = list()
        # Backward propogation through the layers
        for index, layer in enumerate(reversed_layers):
            
            if index > 0:
                loss_grad *= self.activation_grad(layer.output)

            dW, db, loss_grad = layer.backward_propagation(loss_grad)
            gradients.append((dW, db))
        self.update(gradients, self.learning_rate)
        return gradients
    
    def update(self, gradients, learning_rate):
        for layer, (dw, db) in zip(reversed(self.layers), gradients):
            layer.update(learning_rate, dw, db)

        
