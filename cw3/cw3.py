import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Zad 1.1
x = np.array([2000, 2002, 2005, 2007, 2010]).reshape((-1, 1))
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression().fit(x, y)

y_pred1 = model.predict(x)
print(f"Predicted response:\n{y_pred1}")

year = 2020
y_pred2 = model.predict([[year]])
print(f"Predicted response:\n{round(y_pred2[0], 3)}")

# Zad 1.2

years = np.array([2012, 2014, 2016, 2018, 2020, 2022, 2023]).reshape((-1, 1))
y_pred3 = model.predict(years)
print(f"Predicted response:\n{y_pred3}")
print("In year 2023 percent will pass 12%")

# Zad 1.3

x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
ax.set_xlabel("Year")
ax.set_ylabel("Percentage")
ax.grid()

scatter, = ax.plot([], [], 'bo')
line, = ax.plot([], [], 'r')

model = LinearRegression()


def animate(frame_num):
    x_data.append(x[frame_num])
    y_data.append(y[frame_num])

    model.fit(x_data, y_data)
    y_pred = model.predict(x)

    scatter.set_data((x_data, y_data))
    line.set_data(x, y_pred)


anim = FuncAnimation(fig, animate, frames=len(x), interval=150, repeat=False)


# plt.show()


# Zad 2.1
class Perceptron:
    def __init__(self, inputs, targets, learning_rate=0.1, epochs=10):
        self.inputs = inputs
        self.targets = targets
        self.weights = np.random.rand(self.inputs.shape[1])
        self.bias = np.random.rand()
        self.lr = learning_rate
        self.epochs = epochs
        self.train_perceptron()

    def __str__(self):
        print(f"Input\tTarget\tPrediction")
        for idx, i in enumerate(self.inputs):
            print(f"{i}\t\t{self.targets[idx]}\t{self.step_function(i)}")
        return ''

    def step_function(self, x):
        fx = np.dot(self.weights, x) + self.bias
        return np.where(fx >= 0, 1, 0)

    def train_perceptron(self):
        for epoch in range(self.epochs):
            for i in range(len(self.inputs)):
                x = self.inputs[i]
                target = self.targets[i]
                output = self.step_function(x)
                error = target - output
                self.weights += self.lr * error * x
                self.bias += self.lr * error

    def output(self):
        out = []
        for i in self.inputs:
            out.append(int(self.step_function(i)))
        return out


# AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets_and = np.array([0, 0, 0, 1])
p1 = Perceptron(X_and, targets_and)

# NOT
X_not = np.array([[0], [1]])
targets_not = np.array([1, 0])
p2 = Perceptron(X_not, targets_not)

# Zad 2.2

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 1, 0])

p3 = Perceptron(inputs, targets)


# Zad 2.3
# xor - lub i nie-i


def xor(x):
    # OR
    targets_or = np.array([0, 1, 1, 1])
    p5 = Perceptron(x, targets_or)

    # NAND
    targets_nand = np.array([1, 1, 1, 0])
    p6 = Perceptron(x, targets_nand, epochs=100)

    gate_1 = p5.output()  # OR
    gate_2 = p6.output()  # NAND
    gate_3 = np.array([gate_1, gate_2]).T

    targets_xor = np.array([0, 1, 1, 0])
    p7 = Perceptron(gate_3, targets_xor)
    print(p7)


xor(inputs)
