import numpy as np

#sigmoid activation function
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

input = 2 #input layer 2 neurons
hidden = 2 #hidden layer 2 neurons
output = 1 #output layer 1 neuron

#initialize weights and biases randomly
w1 = np.random.randn(input, hidden)
b1 = np.zeros((1, hidden))

w2 = np.random.randn(hidden, output)
b2 = np.zeros((1, output))

#forward propagartion
def forward_prop(X, w1, b1, w2, b2):
    Z1 = np.dot(X, w1) + b1 #weighted sum
    A1 = sigmoid(Z1) #activation function

    Z2 = np.dot(A1, w2) + b2 #weighted sum
    A2 = sigmoid(Z2) #activation function

    return A2


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

outputs = forward_prop(X, w1, b1, w2, b2)


print("outputs:")
print(outputs)