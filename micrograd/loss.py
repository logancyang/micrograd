import numpy as np
from micrograd.engine import Value
from functools import partial

def compute_loss(X, y, model, loss_type, batch_size=None, reg=False):
    # Make batch
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    ValueX = partial(Value, name="X")
    ValueY = partial(Value, name="y")
    inputs = [list(map(ValueX, xrow)) for xrow in Xb]
    yb = ValueY(yb)

    # Forward the model to get scores
    scores = list(map(model, inputs))

    # Compute loss. Specify type of loss, e.g. svm, log, mse
    data_loss = _loss_fn(yb, scores, loss_type)

    # L2 regularization
    alpha = 1e-4
    total_loss = data_loss
    if reg:
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss
    total_loss.name = 'loss'

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0)
                for yi, scorei in zip(yb.data, scores)]

    return total_loss, sum(accuracy) / len(accuracy)


def _loss_fn(yb, scores, loss_type, epsilon=1e-05):
    if loss_type == 'svm':
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    elif loss_type == 'mse':
        losses = [(scorei - yi) * (scorei - yi)
                  for yi, scorei in zip(yb.data, scores)]
    elif loss_type == 'log':  # binary cross entropy
        losses = [scorei.cross_entropy(yi) for yi, scorei in zip(yb.data, scores)]

    data_loss = sum(losses) * (1.0 / len(losses))
    return data_loss
