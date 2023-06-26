from torch.distributions import biject_to
from torch.nn import Module


class Constraint(Module):
    def __init__(self, support, initial_value=None):
        super().__init__()
        self._support = support
        self._transform = biject_to(support)
        self._inv_transform = self._transform.inv
        self.initial_value = initial_value

    def transform(self, tensor):
        return self._transform(tensor)

    def inverse_transform(self, tensor):
        return self._inv_transform(tensor)

    def check_raw(self, tensor):
        return self._support.check(self._transform(tensor)).all()

    def __repr__(self):
        return self._support.__repr__()