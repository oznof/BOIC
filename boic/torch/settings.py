import os

import numpy as np
import torch
import torch.backends.cudnn

from boic.core import Settings
from boic.torch.tests import TorchTestFn, TORCH_TEST_FN_CHOICES


class TorchSettings(Settings):
    DEFAULT_PATH = os.path.join(Settings.DEFAULT_PATH, 'torch')
    ########################################################################################################
    ##### PLATFORM DEPENDENT CLASSES AND METHODS
    # NOTE: other packages need to override these, e.g. torch, tensorflow, jax, ...
    PKG = torch
    RNG_CLS = torch.Generator
    EVAL_FN_CLS = TorchTestFn
    EVAL_FN_DICT = TORCH_TEST_FN_CHOICES

    @property
    def pkg(self):
        return torch

    @property
    def rng_cld(self):
        return self.RNG_CLS

    @property
    def device(self):
        return self['eval']['device']

    @property
    def array_cls(self):
        return torch.Tensor

    def array(self, x, *args, **kwargs):
        if not kwargs:
            kwargs = self.array_options
        return torch.tensor(x, *args, **kwargs)

    def as_array(self, x, *args, **kwargs):
        if not kwargs:
            kwargs = self.array_options
        return torch.as_tensor(x, *args, **kwargs)

    @property
    def array_options(self):
        return dict(dtype=torch.get_default_dtype(), device=self.device)

    @classmethod
    def to_numpy(cls, x, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            return np.asarray(x.detach().clone().cpu().double().numpy(), *args, **kwargs)
        return np.array(x, *args, **kwargs)

    @property
    def default_eval_device_type(self):
        try:
            return self['eval']['device'].type
        except KeyError:
            self.packages_initialize()
            return self['eval']['device'].type

    def packages_initialize(self):
        super().packages_initialize()
        if torch.cuda.is_available():
            self['eval']['device'] = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self['eval']['device'] = torch.device('cpu')
            torch.set_default_tensor_type('torch.DoubleTensor')
        # NOTE: A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater,
        # torch.mm, torch.mv, torch.bmm
        # see: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def rng_initialize(self, seed=None):
        seed = super().rng_initialize(seed)
        torch.manual_seed(seed)
        return seed