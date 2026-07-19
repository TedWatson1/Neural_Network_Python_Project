import numpy as np

class Layer:
    def __init__(self, n_inputs, n_outputs, activation):
        # 'relu', 'leakyrelu', 'linear', 'sigmoid', 'tanh', 'softplus', 'elu'
        match activation:
            case 'relu':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
            case 'leakyrelu':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / ((1 + 0.01 ** 2) * n_inputs))
            case 'linear':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(1/n_inputs)
            case 'sigmoid':
                self.W = np.random.randn(n_inputs, n_outputs) * 4 * np.sqrt(1/n_inputs)
            case 'tanh':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(1/n_inputs)
            case 'softplus':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(1.45/n_inputs)
            case 'elu':
                self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2/n_inputs)
                
        self.b = np.zeros((1, n_outputs))
        self.input = None
        self.output = None


    def forward_pass(self, inputs):
        self.input = inputs
        z1 = inputs @ self.W + self.b
    
        self.output = z1
        return z1
    

    def backward_propagation(self, prev_grad):
        dL_dw = self.input.T @ prev_grad
        dL_db = prev_grad.sum(axis=0, keepdims=True)
        next_grad = prev_grad @ self.W.T

        return dL_dw, dL_db, next_grad

    def update(self, learning_rate, dL_dw, dL_db):
        self.W -= learning_rate * dL_dw
        self.b -= learning_rate * dL_db
