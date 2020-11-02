import torch
import torchvision
import matplotlib.pyplot as plt
import math

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28,
                                   28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
# print(len(x_train_batches))
y_train_batches = torch.split(y_train, batches)


class Mnist:
    def __init__(self):
        # Model variables, initialized using std = 1 / math.sqrt(number of input elements/features).
        # For those interested, read more on normalization and initialization for instance here:
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

        std = 1 / math.sqrt(1 * 5 * 5)
        self.W1 = (torch.randn(32, 1, 5,
                               5) * std).requires_grad_()  # Shape: (out_channels, in_channels, image height, image width)
        self.b1 = (torch.randn(32) * std).requires_grad_()

        std = 1 / math.sqrt(32 * 14 * 14)
        self.W2 = (torch.randn(32 * 14 * 14,
                               10) * std).requires_grad_()  # Shape: (in_channels * image_pixels_after_max_pool, classes)
        self.b2 = (torch.randn(10) * std).requires_grad_()

    def logits(self, x):
        x = torch.nn.functional.conv2d(x, self.W1, self.b1, padding=2)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return x.reshape(-1, 32 * 14 * 14) @ self.W2 + self.b2

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = Mnist()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W1, model.b1, model.W2, model.b2], 0.001)

for batch in range(len(x_train_batches)):
    model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step
print("accuracy = %s" % model.accuracy(x_test, y_test))

# Save pictures
for epoch in range(10):
    # Show the input of the first observation in the training set
    plt.imshow(x_train[epoch, :].reshape(28, 28))

    # Print the classification of the first observation in the training set
    # print(y_train[epoch, :])
    # plt.show()

    # Save the input of the first observation in the training set
    plt.imsave('x_train_' + str(epoch) + '.png', x_train[epoch, :].reshape(28, 28))


