# %%
# from micrograd.engine import Value
# from micrograd.trace_graph import draw_dot

# w = Value(3.0, name='w')
# x = Value(-4.0, name='x')
# b = Value(2.0, name='b')
# y = w*x + b
# y.set_name('y')
# print("y:", y)
# y.backward()

# draw_dot(y)

# %%
# MSELoss
# from micrograd.engine import Value
# from micrograd.trace_graph import draw_dot

# w = Value(3.0, name='w')
# x = Value(-4.0, name='x')
# y = Value(2.0, name='y')
# l = (w*x - y)**2
# l.name = 'MSEloss'
# print("l:", l)
# l.backward()

# draw_dot(l).render('test0')
# %%
# LOGloss
import math
from micrograd.engine import Value
from micrograd.trace_graph import draw_dot

w = Value(3.0, name='w')
x = Value(-4.0, name='x')
y = Value(1.0, name='y')
# Assume True y = 1
dot = Value(w.data * x.data, name='dotprod')
print("dot:", dot)
print("dot.sigmoid():", dot.sigmoid())
l = dot.cross_entropy(1)
# l.name = 'LOGloss'
print("loss:", l)
l.backward()

draw_dot(l).render('test0')

# %%
import numpy as np
from micrograd.engine import Value
from micrograd.trace_graph import draw_dot
from micrograd.nn import Neuron, Layer, MLP
from micrograd.loss import compute_loss

N_PTS = 4
SEED = 1
RADIUS = 1

def f(x):
    return -2*x + 5

def generate_2d_data(n_pts, seed, func=None):
    np.random.seed(seed)
    # Linear regression data
    if func:
        e = np.random.normal(0, 1, n_pts)
        x = np.linspace(-10, 10, n_pts)
        y = f(x) + e
        print("e: ", e)
        return x[:, None], y
    # Binary classification data
    print("Generating classification data...")
    X, y = [], []
    # y = 0 case, x around (3,3)
    for i in range(N_PTS//2):
        pt = np.array([1, 1]) + np.random.normal(0, RADIUS, 2)
        X.append(pt)
        y.append(0)
    for i in range(N_PTS//2):
        pt = np.array([9, 9]) + np.random.normal(0, RADIUS, 2)
        X.append(pt)
        y.append(1)
    return np.array(X), np.array(y)

data = generate_2d_data(n_pts=N_PTS, seed=SEED)
Xs, ys = data
print("Xs", Xs)
print("ys", ys)

# MLP(dim_in, [1]) where [1] is the output neuron.
# This above is a logistic regression
# e.g. MLP(dim_in, [3, 1]) is a MLP with 1 hidden layer which has 3 neurons
# Note: for the 2nd MLP, under 10 pts is too few it won't train. Logreg trains
# fine in this case!
logreg = MLP(2, [1], seed=SEED)
print(logreg)

# draw_dot(total_loss).render('test1')
print("# params:", len(logreg.parameters()))
# update (sgd)
learning_rate = 1e-2
for k in range(1):
    # forward
    total_loss, acc = compute_loss(X=Xs, y=ys, model=logreg, loss_type='log')

    # backward
    logreg.zero_grad()
    total_loss.backward()
    # draw_dot(total_loss).render('test2')
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}")
        print(f"step {k} accuracy {acc*100}%")
    for p in logreg.parameters():
        p.data -= learning_rate * p.grad
        # print(f"grad: {p.grad}")

# if k % 10 == 0:
#     print(f"step {k} loss {total_loss.data}")

# frames[0]
# %%
# frames[0].view()
# linreg.parameters()

# %%
