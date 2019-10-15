import math
import random

class Neural():
    def __init__(self, X, D, learning_rate):
        self.dimension = len(X[0])
        self.data_num = len(D)
        self.weight = []
        for i in range(self. dimension):
            self.weight.append(random.random())
        self.X = X
        self.D = D
        self.learning_rate = learning_rate
        self.bias = -1
        self.epoch = 0
        self.learning_cycle = 0
        self.net = 0

    def Y(self, inputs):
        self.net = 0
        for i in range(len(self.weight)):
            self.net += inputs[i] * self.weight[i]
        self.net -= self.bias
        return self.sgn(self.net)

    def sgn(self, x):
        if x > 0:
            return 1
        else:
            return 0

        #sigmoid
        #return 1 / (1 + math.exp(-x))

    def train(self):
        correct_num = 0
        for i in range(self.data_num):
            print("train: {}: {}".format(i, self.X[i]))
            
            output = self.Y(self.X[i])
            print("output: {}  expect: {}".format(output, self.D[i]))
            print("net: {}".format(self.net))
            print("weight before train: {}, bias before train: {}".format(self.weight, self.bias))
            # same w(n+1) == w(n)
            if output == self.D[i]:
                correct_num += 1
            elif self.net >= 0:
                # adjust weight
                for j in range(len(self.weight)):
                    self.weight[j] -= self.learning_rate* self.X[i][j]
                # adjust bias
                self.bias -= self.learning_rate * -1
            elif self.net < 0:
                # adjust weight
                for j in range(len(self.weight)):
                    self.weight[j] += self.learning_rate* self.X[i][j]
                # adjust bias
                self.bias += self.learning_rate * -1
            print("weight after train: {}, bias after train: {}".format(self.weight, self.bias))
        print("############")
        print("acc: {}".format(correct_num/self.data_num))
        print("############")