# Imports
import math
import random
from collections import Counter
from copy import deepcopy
import time
import statistics

# Pseudo-log
def pseudo_ln(x):
    e = 0.000001
    if x >= e:
        return math.log(x)
    if x < e:
        return math.log(e) + ((x - e) / e)


# Activation functions
def linear(a):
    return a

def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0
    return

def tanh(a):
    return math.tanh(a)

def softsign(a):
    return a / (1 + abs(a))
    
def sigmoid(a):
    if a <= -100:
        return 0
    else:
        return 1 / (1 + math.exp(-a))

def softplus(a):
    if a >= 1000:
        return a
    else:
        return math.log(1 + math.exp(a))

def relu(a):
    #return max([0, a]) * -1
    if a > 0:
        return a
    else:
        return 0


# Loss funcitons
def mean_squared_error(yhat, y):
    return (yhat - y)**2
    
def mean_absolute_error(yhat, y):
    return abs(yhat - y)        

def hinge(yhat, y):
    return max([1 - (yhat*y), 0])

def categorical_crossentropy(yhat, y):
    return -y * pseudo_ln(yhat)
    
def binary_crossentropy(yhat, y):
    return -y * pseudo_ln(yhat) - (1 - y) * pseudo_ln(1 - yhat)


# Derivative
def derivative(function, delta=0.0001):

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
    
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


# Perceptron
class Perceptron():
    # __repr__ and __init__
    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text
        
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0 for i in range(dim)]
        return
    
    def predict(self, xs):
        yhat = []
        for i in range(len(xs)):
            a = self.bias
            # sgn( b + sum( w * x ) )
            # copysign if > 0 = +1, if < 0 = -1, else 0
            for d in range(self.dim):
                a += self.weights[d] * xs[i][d]
            if a > 0:
                yhat.append(1)
            elif a < 0:
                yhat.append(-1)
            else:
                yhat.append(0)
                    
        return yhat

    def partial_fit(self, xs, ys):
        y_hats = self.predict(xs)
        for x, y, y_hat in zip(xs, ys, y_hats):
            # Update hier het perceptron met één instance {x, y}
            # b <- b - (y_hat - y)
            e = y_hat - y
            self.bias -= e
            for i in range(self.dim):
                self.weights[i] -= e * x[i]
        return
    
    def fit(self, xs, ys, epochs=0):
        converge = False
        
        if epochs == 0:
            while converge == False:
                self.partial_fit(xs, ys)
                if self.predict(xs) == ys:
                    converge = True
            
        for epochs in range(epochs):
            self.partial_fit(xs, ys)
        return


# LinearRegression
class LinearRegression():

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text
        
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0 for i in range(dim)]
        return
    
    def predict(self, xs):
        yhat = []
        for i in range(len(xs)):
            a = self.bias
            for d in range(self.dim):
                a += self.weights[d] * xs[i][d]
            yhat.append(a)
        return yhat

    def partial_fit(self, xs, ys, alpha=0.001):
        y_hats = self.predict(xs)
        for x, y, y_hat in zip(xs, ys, y_hats):
            e = alpha * (y_hat - y)
            self.bias -= e
            for i in range(self.dim):
                self.weights[i] -= e * x[i]
        return
    
    def fit(self, xs, ys, alpha=0.001, epochs=0):
        if epochs is 0:
            self.partial_fit(xs, ys)
        else:
            for epochs in range(epochs):
                self.partial_fit(xs, ys, alpha)
        return


class Neuron():
    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text
    
    def __init__(self, dim, activation=linear, loss=mean_absolute_error):
        self.dim = dim
        self.bias = 0
        self.weights = [0 for i in range(dim)]
        self.activation = activation
        self.loss = loss
        return
        
    def predict(self, xs):
        yhat = []
        for i in range(len(xs)):
            a = self.activation(self.bias + sum([self.weights[d] * xs[i][d] for d in range(self.dim)]))
            yhat.append(a)
        return yhat
    
    def partial_fit(self, xs, ys, alpha=0.001):
        # b <- b - a * derivative(self.loss, alpha) * derivative(self.activation, alpha)
        # w <- w - a * derivative(self.loss, alpha) * derivative(self.activation, alpha) * xi
        
        y_hats = self.predict(xs)

        for x, y, y_hat in zip(xs, ys, y_hats):
            d_loss = derivative(self.loss)
            d_act = derivative(self.activation)
            
            self.bias -= alpha * d_loss(y_hat, y) * d_act(self.bias)
            for d in range(self.dim):
                self.weights[d] -= alpha * d_loss(y_hat, y) * d_act(x[d]) * x[d]
        return
    
    def fit(self, xs, ys, alpha=0.001, epochs=100):
        converge = False
        
        if epochs == 0:
            while converge == False:
                self.partial_fit(xs, ys)
                if self.predict(xs) == ys:
                    converge = True
            
        for epochs in range(epochs):
            self.partial_fit(xs, ys)
        return

# Super Class Layer
class Layer():

    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next
        
    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text
    
    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result
        
    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')
    
    def __call__(self, xs, ys=None):
        raise NotImplementedError('Abstract __call__ method')
    
    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs
        
        self.weights = [[(random.uniform(-math.sqrt(6 / (inputs + self.outputs)), math.sqrt(6 / (inputs + self.outputs)))) for i in range(inputs)] for i in range(self.outputs)]
        
        
        
# Input Layer (Child of Layer)
class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text
    
    def __call__(self, xs, ys=None, alpha=None):
        yhats, ls, qs = self.next(xs, ys, alpha)
        return yhats, ls, qs
        
    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean
    
    def partial_fit(self, xs, ys, alpha=0.001, batch_size=0):
        if batch_size > 0:
            main_ls = []
            for b in range(0, len(xs), batch_size):
                sub_xs = xs[b:b + batch_size]
                sub_ys = ys[b:b + batch_size]
                _, ls, _ = self(sub_xs, sub_ys, alpha)
                main_ls += ls
            return main_ls
        else:
            _, ls, _ = self(xs, ys, alpha)
            return ls
    
    def fit(self, xs, ys, epochs=200, alpha=0.001, validation_data=None, batch_size=0):
        history = {'loss': []}
        
        if validation_data is not None:
            history['val_loss'] = []
            
        for e in range(epochs):
            if batch_size > 0:
                cur_time = time.time()
                random.seed(cur_time)
                random.shuffle(xs)
                random.seed(cur_time)
                random.shuffle(ys)
                
            ls = self.partial_fit(xs, ys, alpha, batch_size=batch_size)
            history['loss'].append(statistics.mean(ls))
        
            if validation_data is not None:
                val_xs = validation_data[0]
                val_ys = validation_data[1]
                history['val_loss'].append(self.evaluate(val_xs, val_ys))
        return history
    
    def set_inputs():
        raise NotImplementedError()


# Dense Layer (Child of Layer)
class Dense(Layer):
    
    def __repr__(self):
        text = f'Dense(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(name=name, next=next, outputs=outputs)
        
        self.bias = [0.0 for i in range(self.outputs)]
        self.weights = [[] for i in range(self.outputs)]
        return
    
    def __call__(self, xs, ys=None, alpha=None):
        hs = []
                
        for x in xs:
            h = []
            for o in range(self.outputs):
                h.append(self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs)))
            hs.append(h)
        
        yhats, ls, qs = self.next(hs, ys, alpha)
        
        if alpha is None:
            gs = None
        else:
            for q, x in zip(qs, xs):
                for o in range(self.outputs):
                    factor = alpha * q[o]
                    self.bias[o] -= factor
                    for i in range(self.inputs):
                        self.weights[o][i] -= factor * x[i]
            
            gs = [[sum(q[o] * self.weights[o][i] for o in range(self.outputs)) for i in range(self.inputs)] for q in qs]
        return yhats, ls, gs


# Activation Layer (Child of Layer)
class Activation(Layer):
    
    def __repr__(self):
        text = f'Activation(activation={self.activation.__name__})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text
        
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        
        self.activation = activation
        self.activation_gradient = derivative(activation)
        return
    
    def __call__(self, xs, ys=None, alpha=None):
        hs = []
        for x in xs:
            h = []
            for o in range(self.outputs):
               h.append(self.activation(x[o]))
            hs.append(h)
            
        yhats, ls, qs = self.next(hs, ys, alpha)
        
        if alpha is None:
            gs = None
        else:
            gs = []
            for x, q in zip(xs, qs):
                gs.append([qi * self.activation_gradient(xi) for xi, qi in zip(x, q)])
        return yhats, ls, gs


# Output Layer (Child of Layer)
class OutputLayer(Layer):

    def __init__(self, loss=mean_squared_error, *, name=None):
        self.loss = loss
        self.name = name
        self.loss_gradient = derivative(loss)
        return

    def __repr__(self):
        text = f'OutputLayer(inputs={self.inputs}, loss={self.loss.__name__}, name={self.name})'
        return text
        
    def add(*args, **kwargs):
        return NotImplementedError()
    
    def __call__(self, xs, ys=None, alpha=None):
        yhats = xs
        if ys is None:
            ls = None
            gs = None
        else:
            ls = []
            for x, y in zip(xs, ys):
                l = sum(self.loss(x[i], y[i]) for i in range(self.inputs))
                ls.append(l)
            
            if alpha is None:
                gs = None
            else:
                gs = []                
                for x, y in zip(xs, ys):
                    gs.append([self.loss_gradient(x[i], y[i]) for i in range(self.inputs)])
        return yhats, ls, gs
        
    def set_inputs(self, inputs):
        self.inputs = inputs
        pass
    
    def add():
        raise NotImplementedError()


# Softmax Layer (Child of Layer)
class Softmax(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        return
    
    def __repr__(self):
        text = f'Softmax(name={self.name}, inputs={self.inputs})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text
    
    def __call__(self, xs, ys=None, alpha=None):
        hs = []
        for x in xs:
            h = []
            exps = []
            for o in range(self.outputs):
                exps.append(math.exp(x[o] - max(x)))
            e_sum = sum(exps)
            for e in exps:
                h.append(e / e_sum)
            hs.append(h)
            
        yhats, ls, qs = self.next(hs, ys, alpha)
        
        if alpha is None:
            gs = None
        else:
            # gs = [[1,1,1,1], [1,1,1,1], [1,1,1,1]]
            # ^y: kansen (output softmax) n = inputs, m = outputs
            # kronecker delta (i == o)
            gs = []
            for yh, q in zip(yhats, qs):
                    gs.append([sum([q[o] * yh[o] * ((i == o) - yh[i]) for o in range(self.outputs)]) for i in range(self.inputs)])
        return yhats, ls, gs
