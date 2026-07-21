import numpy as np
from layer import Layer
import math

class Network:
    def __init__(self, num_neurons_per_layer, num_layers, learning_rate, n_inputs, n_outputs, activation_function='relu', output_activation_function='sigmoid'):
        if activation_function.lower().replace(' ', '') not in ['relu', 'leakyrelu', 'linear', 'sigmoid', 'tanh', 'softplus', 'elu']:
            raise ValueError('Activation Function Not Recognised, maybe try:\nRelu, Leaky Relu, Linear, Sigmoid, Tanh, SoftPlus or ELU')
        if output_activation_function.lower().replace(' ', '') not in ['relu', 'leakyrelu', 'linear', 'sigmoid', 'tanh', 'softplus', 'elu']:
            raise ValueError('Output Activation Function Not Recognised, maybe try:\nRelu, Leaky Relu, Linear, Sigmoid, Tanh, SoftPlus or ELU')
        
        self.loss = math.inf

        self.activation_function = activation_function.lower().replace(' ', '')
        self.output_activation_function = output_activation_function.lower().replace(' ', '')
        self.neuron_numbers = num_neurons_per_layer

        # Number of inputs, Number of outputs
        self.layers = [Layer(n_inputs, num_neurons_per_layer, self.activation_function)]
        self.layers.extend([Layer(num_neurons_per_layer, num_neurons_per_layer, self.activation_function) for _ in range(num_layers)])
        self.layers.append(Layer(num_neurons_per_layer, n_outputs, self.output_activation_function))

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
    
    def get_loss(self):
        return self.loss

    def forward(self, x):
        inputs = x
        for index, layer in enumerate(self.layers):
            inputs = layer.forward_pass(inputs)
            if index != len(self.layers) - 1:
                inputs = self.activation(inputs)
            else:
                match self.output_activation_function:
                    case 'relu':
                        inputs = np.maximum(0, inputs)
                    
                    case 'leakyrelu':
                        inputs = np.where(inputs > 0, inputs, inputs * 0.01)
                        
                    case 'linear':
                        pass
                    
                    case 'sigmoid':
                        inputs = 1/(1 + np.exp(-inputs))
                    
                    case 'tanh':
                        inputs = (2/(1 + np.exp(-2*inputs))) - 1
                    
                    case 'softplus':
                        inputs = np.log(1 + np.exp(inputs))
                    
                    case 'elu':
                        inputs = np.where(inputs > 0, inputs, np.exp(inputs) - 1)
        return inputs
    
    
    def backward(self, y_pred, y_true):
        loss_grad = (y_pred - y_true)
        loss_grad *= 2/y_true.size
        self.loss = np.mean((y_pred - y_true) ** 2)
        reversed_layers = list(reversed(self.layers))
        gradients = list()
        # Backward propogation through the layers
        for index, layer in enumerate(reversed_layers):
            
            if index > 0:
                loss_grad *= self.activation_grad(layer.output)
            else:
                match self.output_activation_function:
                    case 'relu':
                        loss_grad *= np.where(layer.output > 0, 1.0, 0.0)
                    
                    case 'leakyrelu':
                        loss_grad *= np.where(layer.output > 0, 1.0, 0.01)
                        
                    case 'linear':
                        loss_grad *= np.ones_like(layer.output)
                    
                    case 'sigmoid':
                        sig = 1.0 / (1.0 + np.exp(-np.clip(layer.output, -500, 500)))
                        loss_grad *= sig * (1.0 - sig)
                    
                    case 'tanh':
                        t = (2/(1 + np.exp(-2*layer.output))) - 1
                        loss_grad *= 1.0 - t ** 2
                    
                    case 'softplus':
                        loss_grad *= 1.0 / (1.0 + np.exp(-np.clip(layer.output, -500, 500)))
                    
                    case 'elu':
                        loss_grad *= np.where(layer.output > 0, 1.0, np.exp(np.clip(layer.output, -500, 500)))

            dW, db, loss_grad = layer.backward_propagation(loss_grad)
            gradients.append((dW, db))
        self.update(gradients, self.learning_rate)
        return gradients
    
    def update(self, gradients, learning_rate):
        for layer, (dw, db) in zip(reversed(self.layers), gradients):
            layer.update(learning_rate, dw, db)

        
