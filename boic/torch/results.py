from boic.core import Method
from .settings import TorchSettings


class TorchMethod(Method):
    BASE_CLS_SETTINGS = TorchSettings
    DEFAULT_EVAL = {'parser': {'E':{'branin': 'Branin', 'rosenb': 'Rosenbrock'}}}
    DEFAULT_ACQ = {'parser': {'NI': {16: ''} }}
