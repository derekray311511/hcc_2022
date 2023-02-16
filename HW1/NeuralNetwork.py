import numpy as np

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        # TODO: return output
        pass

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # def sigmoid(x):
        #     if x >= 0:
        #         return 1 / (1 + np.exp(-x))
        #     else:
        #         return np.exp(x) / (1 + np.exp(x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        relu = lambda x: x*(x > 0)
        relu_prime = lambda x: 1.0*(x > 0)
        super().__init__(relu, relu_prime)

class leakyReLu(Activation):
    def __init__(self):
        leaky_relu = lambda x: x*(x > 0)
        leaky_relu_prime = lambda x: 1.0*(x > 0) + 0.01*(x < 0)
        super().__init__(leaky_relu, leaky_relu_prime)

class Softmax(Activation):
    def __init__(self):
        softmax = lambda x: np.exp(x / np.max(x)) / sum(np.exp(x / np.max(x)))
        softmax_prime = lambda x: (np.exp(x / np.max(x)) / sum(np.exp(x / np.max(x)))) * (1 - np.exp(x / np.max(x)) / sum(np.exp(x / np.max(x))))
        super().__init__(softmax, softmax_prime)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def crossEntropy(y_true, y_pred):
    return -np.sum(np.sum(y_true*np.log(y_pred), axis=1).tolist())

def crossEntropy_prime(y_true, y_pred):
    return -np.sum(np.sum((y_true / y_pred), axis=1).tolist())

def binaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Prevent from overflow in log
    return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def binaryCrossEntropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Prevent from overflow in division
    return -np.sum(y_true / (y_pred) - (1-y_true) / (1-y_pred))

# Use for mini-batch
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def accuracy(y_true, y_pred):
    count = 0 
    max = len(y_true)
    for i in range(max):
        if y_true[i] == (y_pred[i] >= 0.5):
            count = count + 1
    return count / max


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i].reshape((-1, 1))
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    # Get prediction from specific layer
    def get_predict(self, input_data, layer_n):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i].reshape((-1, 1))
            count = 0
            for layer in self.layers:
                output = layer.forward(output)
                if layer_n == count:
                    break
                count += 1
            result.append(output)

        return result

    # train the network
    def fit(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, learning_rate):
        # Error history
        error_hist = []
        test_error_hist = []

        # training loop
        for e in range(epochs):
            error = 0
            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                x_batch, y_batch = batch
                grad = 0
                for x, y in zip(x_batch, y_batch):
                    # forward
                    output = x.reshape((-1, 1))
                    for layer in self.layers:
                        output = layer.forward(output)
                    # error
                    error += self.loss(y, output)
                    # backward
                    grad += self.loss_prime(y, output)

                grad = grad / batch_size
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            # Train error
            error /= len(X_train)
            error_hist.append(error)
            print('Epoch %d/%d,      error = %f' % (e + 1, epochs, error))

            # Validation error
            test_error = 0
            for x, y in zip(X_test, Y_test):
                output = x.reshape((-1, 1))
                for layer in self.layers:
                    output = layer.forward(output)
                test_error += self.loss(y, output)
            test_error /= len(X_test)
            test_error_hist.append(test_error)
            # print('Epoch %d/%d, test_error = %f' % (e + 1, epochs, test_error))

        return error_hist, test_error_hist

    def fit_classification(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, learning_rate):
        # Error history
        error_hist = []
        test_error_hist = []
        trainAcc_hist = []
        testAcc_hist = []

        # training loop
        for e in range(epochs):
            error = 0
            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                x_batch, y_batch = batch
                grad = 0
                for x, y in zip(x_batch, y_batch):
                    # forward
                    output = x.reshape((-1, 1))
                    for layer in self.layers:
                        output = layer.forward(output)
                    # error
                    error += self.loss(y, output)
                    # backward
                    grad += self.loss_prime(y, output)

                grad = grad / batch_size
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            # Train error rate
            error /= len(X_train)
            Y_pred = self.predict(X_train)
            train_acc = accuracy(Y_train, Y_pred)
            error_hist.append(error)
            trainAcc_hist.append(train_acc)
            print('Epoch %d/%d, error = %f, train_acc = %f' % (e + 1, epochs, error, train_acc))

            # Validation error
            test_error = 0
            for x, y in zip(X_test, Y_test):
                output = x.reshape((-1, 1))
                for layer in self.layers:
                    output = layer.forward(output)
                test_error += self.loss(y, output)
            test_error /= len(X_test)
            test_error_hist.append(test_error)

            Ytest_pred = self.predict(X_test)
            test_acc = accuracy(Y_test, Ytest_pred)
            testAcc_hist.append(test_acc)
            # print('Epoch %d/%d, test_error = %f, test_acc = %f' % (e + 1, epochs, test_error, test_acc))

        return error_hist, test_error_hist, trainAcc_hist, testAcc_hist
        
