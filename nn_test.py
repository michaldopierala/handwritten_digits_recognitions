import tensorflow as tf       
import matplotlib.pyplot as plt     
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(' ')
print('*************************************************************************************')
print('*************************************************************************************')
print('***  There is 60 000 samples in data set and you can select index in this range   ***')
print('*************************************************************************************')
print('*************************************************************************************')


# Show lables and indexes
lables = y_train[0:9]
print('This is small sample of data from y_train and its indexes: ')
print(f'Index : {np.arange(len(lables))}')
print(f'Lables: {lables}')


# Select an index
index = input('Enter the index of the nunber: ')
index = int(index)

# Select the image at the specified index
selected_image = x_train[index]

# Display the image
print('In order to continue close the window with displayed number')
plt.imshow(selected_image, cmap='gray')
plt.title(f"Label: {y_train[index]}")
plt.show()


# NN gues
W = np.load('weights.npy', allow_pickle=True)
B = np.load('biases.npy', allow_pickle=True)

print(f'Number selected from y_train: {y_train[index]}')


number = x_train[index]
number = (number / 255.0).reshape(-1, 784).T.astype(np.float32)

def softmax(z):
    Z_max = np.max(z, axis=0, keepdims=True)
    Z_shifted = z - Z_max
    exp_vals = np.exp(Z_shifted)
    sum_exp_vals = np.sum(exp_vals, axis=0, keepdims=True)
    A = exp_vals / sum_exp_vals
    return A

def relu(X):
    return np.maximum(0,X)

def forward_propagation(number, W, B):
    activations = [number]
    zs = []
    for i in range(len(W)):
        z = np.dot(W[i], activations[i]) + B[i]
        zs.append(z)
        if i == len(W) - 1:
            activations.append(softmax(z))
        else:
            activations.append(relu(z))
    return activations[-1]


result = forward_propagation(number, W, B)
result = np.around(result, 4)
print(' ')
print('The result from forward propagation function: ')
print(result.T)
print(' ')
print( 'Index of max value(NN guess):')
print(np.argmax(result))