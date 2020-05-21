import numpy as np
from micrograd.engine import Value


def compute_loss(X, y, model, loss_type, batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # Specify type of loss, e.g. svm, log, mse
    data_loss = _loss_fn(yb, scores, loss_type)

    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0)
                for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def _loss_fn(yb, scores, loss_type, epsilon=1e-05):
    if loss_type == 'svm':
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    elif loss_type == 'mse':
        losses = [(scorei - yi) * (scorei - yi)
                  for yi, scorei in zip(yb, scores)]
    elif loss_type == 'log':  # cross entropy
        losses = [(yi * np.log(scorei + epsilon) +
                   (1 - yi) * np.log(1 - scorei + epsilon))
                  for yi, scorei in zip(yb, scores)]

    data_loss = sum(losses) * (1.0 / len(losses))
    return data_loss
