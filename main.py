
from model import *

net = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

for epoch in range(20):

    ypred = [net(x) for x in xs]

    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    net.zero_grad()

    for p in net.parameters():
        p.grad = 0.0

    loss.backward()

    for p in net.parameters():
        p.data += -0.01 * p.grad

    print(loss.data)
