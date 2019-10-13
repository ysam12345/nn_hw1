import math
import random


class Neural():
    def __init__(self, X, D, learning_rate):
        self.dimension = len(X[0])
        self.data_num = len(D)
        self.weight = []
        for in range(dimension)
            self.weight.append(random.random())
        self.X = X
        self.D = D
        self.learning_rate = learning_rate
        self.bias = -1
        self.epoch = 0
        self.learning_cycle = 0
        self.net = 0

        

    def get_weight(self):
        return self.weight
    
    def get_bias(self):
        return self.bias

    def Y(self, input):
        for i in range(len(self.weight)):
            self.net += self.input[i] * self.weight[i]
        self.net += bias
        return sigmoid(self.net)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self):
        for i in range(self.data_num):
            print("train: {}: {}".format(i, X[i]))
            # same w(n+1) == w(n)
            if self.Y(X[i]) == self.D[i]:
                pass
            elif self.Y(X[i]) == self.D[i]