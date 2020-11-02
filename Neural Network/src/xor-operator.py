import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d, art3d

matplotlib.rcParams.update({'font.size': 11})

# Regarding the notations, see http://stats.stackexchange.com/questions/193908/in-machine-learning-why-are-superscripts-used-instead-of-subscripts

# W1_init = torch.tensor([[7.43929911, 5.68582106], [7.44233704, 5.68641663]], requires_grad=True)
# b1_init = torch.tensor([[-3.40935969, -8.69532299]], requires_grad=True)
# W2_init = torch.tensor([[13.01280117], [-13.79168701]], requires_grad=True)
# b2_init = torch.tensor([[-6.1043458]], requires_grad=True)

# Also try:
W1_init = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
b1_init = torch.tensor([[-5.0, 15.0]], requires_grad=True)
W2_init = torch.tensor([[10.0], [10.0]], requires_grad=True)
b2_init = torch.tensor([[-15.0]], requires_grad=True)


#def sigmoid(t):
#    return 1 / (1 + np.exp(-t))


class SigmoidModel:
    def __init__(self, W1=W1_init, W2=W2_init, b1=b1_init, b2=b2_init):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    # First layer function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return -torch.mean(torch.mul(y, torch.log(self.f(x))) + torch.mul((1 - y), torch.log(1 - self.f(x))))


model = SigmoidModel()

# Observed/training input and output
x_train = torch.tensor([0., 0., 0., 1., 1., 0., 1., 1.]).reshape(-1, 2)
y_train = torch.tensor([0., 1., 1., 0.]).reshape(-1, 1)

optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step
    #print(model.W, model.b)

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(121, projection='3d')

plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$h=$f1$(x)=\\sigma(x$W1$+$b1$)$")
plot1_h1 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))
plot1_h2 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))

plot1.plot(x_train[:, 0].squeeze(),
           x_train[:, 1].squeeze(),
           y_train[:, 0].squeeze(),
           'o',
           label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
           color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$h_1,h_2$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

plot2 = fig.add_subplot(222, projection='3d')

plot2_f2 = plot2.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=$f2$(h)=\\sigma(h $W2$+$b2$)$")

plot2_info = fig.text(0.8, 0.9, "")

plot2.set_xlabel("$h_1$")
plot2.set_ylabel("$h_2$")
plot2.set_zlabel("$y$")
plot2.legend(loc="upper left")
plot2.set_xticks([0, 1])
plot2.set_yticks([0, 1])
plot2.set_zticks([0, 1])
plot2.set_xlim(-0.25, 1.25)
plot2.set_ylim(-0.25, 1.25)
plot2.set_zlim(-0.25, 1.25)

plot3 = fig.add_subplot(224, projection='3d')

plot3_f = plot3.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=f(x)=$f2$($f1$(x))$")

plot3_info = fig.text(0.3, 0.03, "")

plot3.set_xlabel("$x_1$")
plot3.set_ylabel("$x_2$")
plot3.set_zlabel("$y$")
plot3.legend(loc="upper left")
plot3.set_xticks([0, 1])
plot3.set_yticks([0, 1])
plot3.set_zticks([0, 1])
plot3.set_xlim(-0.25, 1.25)
plot3.set_ylim(-0.25, 1.25)
plot3.set_zlim(-0.25, 1.25)

table = plt.table(cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                  cellLoc="center",
                  loc="lower right")


def update_figure(event=None):
    if (event is not None):
        if event.key == "W":
            model.W1[0, 0] += 0.2
        elif event.key == "w":
            model.W1[0, 0] -= 0.2
        elif event.key == "E":
            model.W1[0, 1] += 0.2
        elif event.key == "e":
            model.W1[0, 1] -= 0.2
        elif event.key == "R":
            model.W1[1, 0] += 0.2
        elif event.key == "r":
            model.W1[1, 0] -= 0.2
        elif event.key == "T":
            model.W1[1, 1] += 0.2
        elif event.key == "t":
            model.W1[1, 1] -= 0.2

        elif event.key == "Y":
            model.W2[0, 0] += 0.2
        elif event.key == "y":
            model.W2[0, 0] -= 0.2
        elif event.key == "U":
            model.W2[1, 0] += 0.2
        elif event.key == "u":
            model.W2[1, 0] -= 0.2

        elif event.key == "B":
            model.b1[0, 0] += 0.2
        elif event.key == "b":
            model.b1[0, 0] -= 0.2
        elif event.key == "N":
            model.b1[0, 1] += 0.2
        elif event.key == "n":
            model.b1[0, 1] -= 0.2

        elif event.key == "M":
            model.b2[0, 0] += 0.2
        elif event.key == "m":
            model.b2[0, 0] -= 0.2

        elif event.key == "c":
            model.W1 = W1_init
            model.W2 = W2_init
            model.b1 = b1_init
            model.b2 = b2_init

    global plot1_h1, plot1_h2, plot2_f2, plot3_f
    plot1_h1.remove()
    plot1_h2.remove()
    plot2_f2.remove()
    plot3_f.remove()
    x1_grid, x2_grid = torch.meshgrid(torch.linspace(-0.25, 1.25, 10), torch.linspace(-0.25, 1.25, 10))
    h1_grid = np.empty([10, 10])
    h2_grid = np.empty([10, 10])
    f2_grid = np.empty([10, 10])
    f_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            h = model.f1(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]]))
            h1_grid[i, j] = h[0, 0]
            h2_grid[i, j] = h[0, 1]
            f2_grid[i, j] = model.f2(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]]))
            f_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]]))

    plot1_h1 = plot1.plot_wireframe(x1_grid, x2_grid, h1_grid, color="lightgreen")
    plot1_h2 = plot1.plot_wireframe(x1_grid, x2_grid, h2_grid, color="darkgreen")

    plot1_info.set_text("W1$=\\left[\\genfrac{}{}{0}{}{%.2f}{%.2f}\\/\\genfrac{}{}{0}{}{%.2f}{%.2f}\\right]$\nb1$=[{%.2f}, {%.2f}]$" %
                        (model.W1[0, 0], model.W1[1, 0], model.W1[0, 1], model.W1[1, 1], model.b1[0, 0], model.b1[0, 1]))

    plot2_f2 = plot2.plot_wireframe(x1_grid, x2_grid, f2_grid, color="green")

    plot2_info.set_text("W2$=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\nb2$=[{%.2f}]$" % (model.W2[0, 0], model.W2[1, 0], model.b2[0, 0]))

    plot3_f = plot3.plot_wireframe(x1_grid, x2_grid, f_grid, color="green")

    plot3_info.set_text(
        "$loss = -\\frac{1}{n}\\sum_{i=1}^{n}\\left [ \\hat y^{(i)} \\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$" %
        model.loss(x_train, y_train))

    table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f(torch.tensor([[0., 0.]])))
    table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[0., 1.]])))
    table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[1., 0.]])))
    table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f(torch.Tensor([[1., 1.]])))
    fig.canvas.draw()


update_figure()
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()
