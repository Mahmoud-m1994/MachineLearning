import matplotlib.pyplot as plt
import pandas as pandas
import torch as torch

# Reading input and output from file without rounding off

with pandas.option_context('display.precision', 20):
    data_from_file = pandas.read_csv("length_weght.csv", float_precision=None)

# creating tensor from targets_df
allData = torch.tensor(data_from_file.values).float()
# print(allData)
# Split allData to x and y train tensors
x_train = torch.reshape(allData[:, 0],(-1,1))
y_train = torch.reshape(allData[:, 1],(-1,1))
# print(x_train_asDouble)

# printing out result
# print(x_train)

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return  torch.mm(x , self.W) + self.b # x  @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)
for epoch in range(1000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
plt.legend()
plt.show()




