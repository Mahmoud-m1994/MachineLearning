import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import torch as torch

# Reading input and output from file without rounding off
with pandas.option_context('display.precision', 20):
    data_from_file = pandas.read_csv("day_length_weight.csv", float_precision=None)

# creating tensor from targets_df
allData = torch.tensor(data_from_file.values).float()
# print(allData)
# Split allData to x and y train tensors
x_train = torch.reshape(allData[:, (0,1)],(-1,2)).numpy()
y_train = torch.reshape(allData[:, 2],(-1,1)).numpy()

matplotlib.rcParams.update({'font.size': 11})

W_init = numpy.array([[-0.2], [0.53]])
b_init = numpy.array([[3.1]])


class Linear3dRegressionModel:

    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = W
        self.b = b

    # predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return numpy.mean(numpy.power(self.f(x) - y, 2))

model = Linear3dRegressionModel()

fig = plt.figure('Linear regression: 3D')

plot1 = fig.add_subplot(111, projection='3d')

plot1.plot(x_train[:, 0].squeeze(),
           x_train[:, 1].squeeze(),
           y_train[:, 0].squeeze(),
           'o',
           label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$',
           color='blue')

plot1_f = plot1.plot_wireframe(numpy.array([[]]), numpy.array([[]]), numpy.array([[]]), color='green', label='$y = f(x) = xW+b$')

plot1_info = fig.text(0.01, 0.02, '')

plot1_loss = []

for i in range(0, x_train.shape[0]):
    line, = plot1.plot([0, 0], [0, 0], [0, 0], color='red')
    plot1_loss.append(line)
    if i == 0:
        line.set_label('$|f(\\hat x^{(i)})-\\hat y^{(i)}|$')

plot1.set_xlabel('$x_1$')
plot1.set_ylabel('$x_2$')
plot1.set_zlabel('$y$')
plot1.legend(loc='upper left')
plot1.set_xticks([])
plot1.set_yticks([])
plot1.set_zticks([])
plot1.w_xaxis.line.set_lw(0)
plot1.w_yaxis.line.set_lw(0)
plot1.w_zaxis.line.set_lw(0)
plot1.quiver([0], [0], [0], [numpy.max(x_train[:, 0] + 1)], [0], [0], arrow_length_ratio=0.05, color='black')
plot1.quiver([0], [0], [0], [0], [numpy.max(x_train[:, 1] + 1)], [0], arrow_length_ratio=0.05, color='black')
plot1.quiver([0], [0], [0], [0], [0], [numpy.max(y_train[:, 0] + 1)], arrow_length_ratio=0.05, color='black')


def update_figure(event=None):
    if (event is not None):
        if event.key == 'W':
            model.W[0, 0] += 0.00001
        elif event.key == 'w':
            model.W[0, 0] -= 0.00001
        elif event.key == 'E':
            model.W[1, 0] += 0.00001
        elif event.key == 'e':
            model.W[1, 0] -= 0.00001

        elif event.key == 'B':
            model.b[0, 0] += 0.025
        elif event.key == 'b':
            model.b[0, 0] -= 0.025

        elif event.key == 'c':
            model.W = W_init.copy()
            model.b = b_init.copy()

    global plot1_f
    plot1_f.remove()
    x1_grid, x2_grid = numpy.meshgrid(numpy.linspace(1, 6, 10), numpy.linspace(1, 4.5, 10))
    y_grid = numpy.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])
    plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color='green')

    for i in range(0, x_train.shape[0]):
        plot1_loss[i].set_data(numpy.array([x_train[i, 0], x_train[i, 0]]), numpy.array([x_train[i, 1], x_train[i, 1]]))
        plot1_loss[i].set_3d_properties(numpy.array([y_train[i, 0], model.f(x_train[i, :])]))

    plot1_info.set_text(
        '$W=\\left[\\stackrel{%.2f}{%.2f}\\right]$\n$b=[%.2f]$\n$loss = \\frac{1}{n}\\sum_{i=1}^{n}(f(\\hat x^{(i)}) - \\hat y^{(i)})^2 = %.2f$' %
        (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))

    fig.canvas.draw()


update_figure()
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()
