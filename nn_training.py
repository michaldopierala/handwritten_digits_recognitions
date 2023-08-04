import tensorflow as tf       
import matplotlib.pyplot as plt     
import numpy as np
from datetime import datetime
import gc

time_start = utc_time = datetime.utcnow()


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# save number to to txt file - this is only for visualisa
# rounded_number = x_train
# rounded_number = np.ceil(rounded_number)
# np.savetxt("data2.txt", rounded_number[6], fmt='%d')


# print("Training data shape:", x_train.shape)  # (60000, 28, 28)
# print("Training labels shape:", y_train.shape)  # (60000,)
# print("Test data shape:", x_test.shape)  # (10000, 28, 28)
# print("Test labels shape:", y_test.shape)  # (10000,)
# print("test lables item --",y_test[3] )


# conver 28x28 data samples into one row of 784 input data 
# x_train = x_train / 255
# convert =[]
# for x in x_train:
#     x= x.flatten()
#     convert.append(x)

# x_train = (np.array(convert))
# x_train = x_train.T

x_train = (x_train / 255.0).reshape(-1, 784).T.astype(np.float16)
n, m =x_train.shape # m should the amount of examples and n should be amount of features +1 (for lables)
print(f'amount of features {n}')
print(f'amount of examples {m}')



class NeuralNetwork:
    def __init__(self, architecture):
        self.biases = [np.random.randn(y, 1) for y in architecture[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(architecture[:-1], architecture[1:])]
        self.activations = []
        self.zs = []


    def softmax(self, z):
        Z_max = np.max(z, axis=0, keepdims=True)
        Z_shifted = z - Z_max
        exp_vals = np.exp(Z_shifted)
        sum_exp_vals = np.sum(exp_vals, axis=0, keepdims=True)
        A = exp_vals / sum_exp_vals
        return A

    def relu(self, X):
        return np.maximum(0,X)

    def relu_derivative(self,X):
        return np.where(X<=0,0,1)


    def one_zer(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T 
        # np.savetxt("Y.txt", one_hot_Y.T , fmt='%f')
        return one_hot_Y

    def forward_propagation(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.zs.append(z)
            if i == len(self.weights) - 1:
                self.activations.append(self.softmax(z))
            else:
                 self.activations.append(self.relu(z))
        #     self.activations.append(self.relu(z))
        # self.activations[-1] = self.softmax(self.activations[-1])
        # np.save('activation.npy', self.activations)
        # np.save('zs.npy', self.zs)
        return self.activations, self.zs





    def backward_propagation(self, X, y):
        y = self.one_zer(y)
        deltas = [None] * len(self.weights)
        deltas[-1] = (self.activations[-1] - y) #* self.sigmoid_derivative(self.activations[-1])
        # print(f'to jest len weights: {len(self.weights)}')
        # for i in range(len(self.weights)- 2, -1, -1):
        #     print(f'to jest i: {i} ') 
        for i in range(len(self.weights)- 2, -1, -1):    # zamienilem -1 na zero 
            deltas[i] = (np.dot(self.weights[i + 1].T, deltas[i + 1]) * self.relu_derivative(self.zs[i]))
        return deltas

    def update_parameters(self, X, y, learning_rate):
        # deltas = self.backward_propagation(X, y)
        deltas = self.backward_propagation(X, y)
        for i in range(len(self.weights)):
            dW = np.dot(deltas[i], self.activations[i].T)
            dB = np.mean(deltas[i], axis=1, keepdims=True)
            self.weights[i] -= learning_rate * 1/m * dW
            self.biases[i] -= learning_rate * 1/m * dB
        # Clear out the activations and zs after using - it reduced significantly load on RAM and SWAP memory
        # It solved problem with SWAP memory overlload in tranning with long epoch. Training is a little bit slower 
        self.activations.clear()
        # self.zs.clear()
        # Collect garbage
        # gc.collect()

    def calculate_accuracy(self, X, Y):
        A, Z = self.forward_propagation(X)
        A= A[-1]
        max = np.argmax(A, 0)
        accuracy = np.sum(max == Y) / Y.size
        return accuracy

    def train(self, X, Y, learning_rate, epochs):
        for epoch in range(epochs):
            # np.save(f'con_{epoch}_W.npy', self.weights)
            # np.save(f'con_{epoch}_A.npy', self.activations)
            # np.save(f'con_{epoch}_Z.npy', self.zs)
            # np.save(f'con_{epoch}_B.npy', self.biases)
            self.forward_propagation(X)
            self.update_parameters(X, Y, learning_rate)
            if epoch % 10 == 0:
              print("Iteration: ", epoch)
              accuracy = self.calculate_accuracy(X, Y)
              print(f'Accuracy: {accuracy}')
        return accuracy, self.weights, self.biases 





epochs = 50
learning_rate = 1
architectur = [784, 600, 10]

nn = NeuralNetwork(architectur)
# f, zs = nn.forward_propagation(x_train)
# accuracy = nn.calculate_accuracy(x_train, y_train)
accuracy, weights, biases   = nn.train(x_train, y_train, learning_rate, epochs )
# np.save(f'weights.npy', weights)
np.save(f'biases.npy', biases )

print(accuracy)
# print(nn.weights)
# print(f'this is the result of nn {zs[-1]}')
# print(f'this is the shape of nn {(len(a) for a in f[-1])}')
# print(f'shapee: {zs[-1][:,0].shape} ')
# print({len(f)})


tiem_finish = datetime.utcnow() - time_start

print(f'total calculation time: {tiem_finish}')