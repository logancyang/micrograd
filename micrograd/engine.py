import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', name='tensor'):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._name = name

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        op = '+'
        out = Value(self.data + other.data, (self, other), op)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        op = '*'
        out = Value(self.data * other.data, (self, other), op)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        op = f'**{other}'
        out = Value(self.data**other, (self,), op)

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        return 1.0/(1 + math.exp(-self.data))

    def cross_entropy(self, target):
        """Binary case"""
        op = "logloss(y)"
        eps = 1e-5
        loss = -(target * math.log(self.sigmoid()+eps) + \
                (1 - target) * math.log((1 - self.sigmoid()+eps)))
        out = Value(loss, (self,), op, name='logloss')
        def _backward():
            self.grad = self.sigmoid() - target
        out._backward = _backward
        return out

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU', name="ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """Backprop, recursively traverse the compute graph.
        Focus on this code in the article. Make a gif for the construction
        of a logistic regression for stepping thru this code.
        """
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, name={self.name})"
