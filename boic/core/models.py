from abc import ABC, abstractmethod
from typing import Dict


class SurrogateModel(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_train_data(self, inputs=None, targets=None, **kwargs):
        # ... handle type conversion
        self.train_inputs = inputs
        self.train_targets = targets
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, info_to_numpy=True, **kwargs):
        # any relevant stats during train should be stored in train_info, keep only current
        self.train_info = None
        raise NotImplementedError

    @abstractmethod
    def state(self, state : Dict = None, copy=False, to_numpy=True) -> Dict:
        # if state is not None, set to state
        # always return current state
        raise NotImplementedError

    @abstractmethod
    def acquire(self, **kwargs):
        raise NotImplementedError