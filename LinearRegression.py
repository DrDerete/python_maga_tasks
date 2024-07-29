import numpy as np


def generate_batches(x, y, batch_size):
    assert len(x) == len(y)
    np.random.seed(42)
    x = np.array(x)
    y = np.array(y)
    perm = np.random.permutation(len(x))
    batch_count = len(y) // batch_size

    for batch_start in range(0, batch_count):
        piece = perm[batch_start * batch_size:batch_start * batch_size + batch_size]
        yield x[piece], y[piece]


def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1. / (1 + np.exp(-h))


class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=3):
        n, k = x.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)
        x_train = np.concatenate((np.ones((n, 1)), x), axis=1)
        losses = []
        for i in range(epochs):
            for X_batch, y_batch in generate_batches(x_train, y, batch_size):
                predictions = np.clip(self.predict(X_batch), 1e-10, 1 - 1e-10)
                loss = self.__loss(y_batch, predictions)
                assert (np.array(loss).shape == tuple()), "Лосс должен быть скаляром!"

                losses.append(loss)

                self.w -= lr * self.get_grad(X_batch, y_batch, sigmoid(logit(X_batch, self.w)))

        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        grad_basic = np.transpose(X_batch) @ (predictions - y_batch)
        assert grad_basic.shape == (X_batch.shape[1],), "Градиенты должны быть столбцом из k_features + A_testing.py элементов"
        return grad_basic

    def _predict_proba_internal(self, x):
        return sigmoid(logit(x, self.w))

    def predict(self, x, threshold=0.5):
        return self._predict_proba_internal(x) >= threshold

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

