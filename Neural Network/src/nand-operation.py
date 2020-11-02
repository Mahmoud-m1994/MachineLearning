import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d, art3d

matplotlib.rcParams.update({'font.size': 11})

# regarding the notations, see http://stats.stackexchange.com/questions/193908/in-machine-learning-why-are-superscripts-used-instead-of-subscripts


W_init = torch.tensor([[0.], [2.]], requires_grad=True)
b_init = torch.tensor([[-3.4]], requires_grad=True)


#def sigmoid(t):
#    return 1 / (1 + np.exp(-t))


class SigmoidModel:
    def __init__(self, W=W_init,b=b_init):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Uses Cross Entropy
    def loss(self, x, y):
        #print(torch.mean(torch.mul(y, torch.log(self.f(x))) + torch.mul((1 - y), torch.log(1 - self.f(x)))))
        return -torch.mean(torch.mul(y, torch.log(self.f(x))) + torch.mul((1 - y), torch.log(1 - self.f(x))))



model = SigmoidModel()

# Observed/training input and output
x_train = torch.tensor([0., 0., 0., 1., 1., 0., 1., 1.]).reshape(-1, 2)
y_train = torch.tensor([1., 1., 1., 0.]).reshape(-1, 1)


optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step
    #print(model.W, model.b)


fig = plt.figure("Logistic regression: the logical OR operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=f(x)=\\sigma(xW+b)$")

plot1.plot(x_train[:, 0].squeeze(),
           x_train[:, 1].squeeze(),
           y_train[:, 0].squeeze(),
           'o',
           label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
           color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

table = plt.table(cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                  cellLoc="center",
                  loc="lower right")


def update_figure(event=None):
    if (event is not None):
        if event.key == "W":
            model.W[0, 0] += 0.2
        elif event.key == "w":
            model.W[0, 0] -= 0.2
        elif event.key == "E":
            model.W[1, 0] += 0.2
        elif event.key == "e":
            model.W[1, 0] -= 0.2

        elif event.key == "B":
            model.b[0, 0] += 0.2
        elif event.key == "b":
            model.b[0, 0] -= 0.2

        #elif event.key == "c":
            model.W = W_init.copy()
            model.b = b_init.copy()

    global plot1_f
    plot1_f.remove()
    x1_grid, x2_grid = torch.meshgrid(torch.linspace(-0.25, 1.25, 10), torch.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.Tensor([[x1_grid[i, j], x2_grid[i, j]]]))
    plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

    plot1_info.set_text(
        "$W=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\n$b=[%.2f]$\n$loss = -\\frac{1}{n}\\sum_{i=1}^{n}\\left [ \\hat y^{(i)} \\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$"
        % (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))

    table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[0, 0]])))
    table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[0, 1]])))
    table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[1, 0]])))
    table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[1, 1]])))

    fig.canvas.draw()


update_figure()
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()
