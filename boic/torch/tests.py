import math
import numpy as np

import torch

from boic.core.tests import TestFn
from .rover_test import rover, srover, mrover


class TorchTestFn(TestFn):
    PKG = torch
    RNG_CLS = torch.Generator
    ARGS = ('dim', 'seed', 'noise', 'bounds', 'device')

    def __init__(self, dim, seed, noise=None, bounds=None, device='cpu', rng_state=None, **kwargs):
        self.device = torch.device(device)
        TestFn.__init__(self, dim=dim, seed=seed, noise=noise, bounds=bounds, rng_state=rng_state, **kwargs)

    @property
    def pkg(self):
        return self.PKG

    @property
    def rng_cls(self):
        return self.RNG_CLS

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
            return np.asarray(x.cpu().double().numpy(), *args, **kwargs)
        return np.array(x, *args, **kwargs)

    def rng_initialize(self, seed=None, state=None):
        if seed:
            self.seed = seed
        self.rng = self.RNG_CLS(device=self.device).manual_seed(self.seed)
        if state is not None:
            self.rng.set_state(state)

    def f(self, inputs):
        raise NotImplementedError

    def f_and_grad(self, inputs):
        inputs = torch.atleast_2d(self.as_array(inputs)).requires_grad_(True)
        loss = self.f(inputs)
        curr_loss = loss.item()
        inputs.grad = torch.autograd.grad([loss], [inputs])[0]
        return curr_loss, inputs.grad.clone().squeeze()

    def eval(self, inputs, grad=False):
        if not grad:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        inputs = torch.atleast_2d(self.as_array(inputs))
        targets = self.f(inputs)
        if self.noise:
            targets += self.generate_noise(targets.shape)
        if not grad:
            torch.set_grad_enabled(prev)
        return targets

    @staticmethod
    def generate_noise_normal(shape, noise=1, rng=None):
        return  math.sqrt(noise) * torch.randn(shape, generator=rng)

    def argbetter(self, x):
        return self.PKG.argsort(x) if self.minimize else self.PKG.argsort(x, descending=True)

    def argworse(self, x):
        return self.PKG.argsort(x, descending=True) if self.minimize else self.PKG.argsort(x)


class Sphere(TorchTestFn):
    ARGS = ('min_input', 'dim', 'seed', 'noise', 'bounds')

    def __init__(self, *args, min_input=None, **kwargs):
        TorchTestFn.__init__(self, *args, **kwargs)
        if min_input is None:
            min_input = self.pkg.ones(self.dim) * 0.42
        self.min_input = self.as_array(min_input).reshape(1, -1)

    def f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        return self.pkg.sum((inputs - self.min_input) ** 2, -1)

    @property
    def best_input_global(self):
        if self._best_input_global is None:
            self._best_input_global = self.min_input
        return self._best_input_global


class GaussianKernels(TorchTestFn):
    ARGS = ('min_input', 'outputscales', 'lengthscales',  'dim', 'seed', 'noise')

    def __init__(self, min_inputs, outputscales, lengthscales,  *args, **kwargs):
        TorchTestFn.__init__(self, *args, **kwargs)
        self.min_inputs = self.as_array(min_inputs).reshape(-1, self.dim)
        self.outputscales = self.as_array(outputscales).reshape(1, -1)
        self.lengthscales = self.as_array(lengthscales).reshape(1, -1)

    def f(self, inputs):
        res = self.outputscales * \
              torch.exp(-torch.cdist(inputs, self.min_inputs).pow(2).div(2 * self.lengthscales ** 2))
        return torch.sum(res, -1)

    @property
    def best_input_global_guess(self):
        return self.min_inputs


class Rosenbrock(TorchTestFn):
    def __init__(self, dim, *args, f_shift=2.5, f_scale=7.5, **kwargs):
        assert (dim >= 2)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.shift = f_shift
        self.scale = f_scale
        self.min_input = (self.as_array(self.pkg.ones(self.dim)) - self.shift) / self.scale
        # the commented formula is the implementation in hypersphere, but it is not completely correct
        # self.normalizer = 50000.0 / ((90 ** 2 + 9 ** 2) * (self.dim - 1))
        self.min_input = self.min_input.reshape(-1, self.dim)
        self.target_rescale = 100.
        # 1. this rescales to [0, 100]
        # self.normalizer = 100 * 105**2 + 9**2 + (100 * 90**2 + 9**2) * (self.dim - 2)
        # self.rescale = self.target_rescale / self.normalizer
        # 2. center value rescaled to 100
        # this makes worst become about 60k-80k (becomes smaller with dim)
        self.rescale = 1
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()
        self.partition = [self.pkg.arange(2) + i for i in range(self.dim)]  # additive with overlapping groups
        # print(self.worst_target_global)
        # print(self.f(self.as_array(self.pkg.zeros(self.dim))).item())

    @property
    def best_input_global(self):
        if self._best_input_global is None:
            self._best_input_global = self.min_input
        return self._best_input_global

    @property
    def worst_input_global(self):
        if self._worst_input_global is None:
            iw = self.as_array(self.pkg.ones(self.dim))
            iw[-1] = - iw[-1]
            self._worst_input_global = iw.reshape(-1, self.dim)
        return self._worst_input_global

    def f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.shift
        return (100.0 * ((x[..., 1:] - x[..., :-1] ** 2) ** 2) + (x[..., :-1] - 1) ** 2).sum(-1) * self.rescale


class SRosenbrock(Rosenbrock):
    def __init__(self, dim, *args, f_shift=-2.75, f_scale=7.5, **kwargs):
        super().__init__(dim, *args, f_shift=f_shift, f_scale=f_scale, **kwargs)

    @property
    def worst_input_global(self):
        return super(Rosenbrock, self).worst_input_global


class SSRosenbrock(Rosenbrock):
    def __init__(self, dim, *args, f_shift=-4.625, f_scale=7.5, **kwargs):
        super().__init__(dim, *args, f_shift=f_shift, f_scale=f_scale, **kwargs)

    @property
    def worst_input_global(self):
        return super(Rosenbrock, self).worst_input_global


class S35Rosenbrock(Rosenbrock):
    def __init__(self, dim, *args, f_shift=-1.625, f_scale=7.5, **kwargs):
        super().__init__(dim, *args, f_shift=f_shift, f_scale=f_scale, **kwargs)

    @property
    def worst_input_global(self):
        return super(Rosenbrock, self).worst_input_global


class S65Rosenbrock(Rosenbrock):
    def __init__(self, dim, *args, f_shift=-3.875, f_scale=7.5, **kwargs):
        super().__init__(dim, *args, f_shift=f_shift, f_scale=f_scale, **kwargs)

    @property
    def worst_input_global(self):
        return super(Rosenbrock, self).worst_input_global


class Branin(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim % 2 == 0)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.num_repeats = dim // 2
        self.shift = self.as_array([2.5, 7.5])
        self.repeated_shift = self.shift.repeat(self.num_repeats)
        self.scale = 7.5
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        # original: min_input = self.as_array([[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]])
        #           min_input = (min_input - self.shift) / self.scale
        min_input = self.array([-0.82525, 0.81739])
        self.min_input = min_input.repeat(1, self.num_repeats)
        self.best_shift_global = 0
        self.target_rescale = 100.
        self.rescale = 1
        self.best_shift_global = self._f(self.best_input_global).min().item()
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    # @property
    # def best_input_global(self):
    #     if self._best_input_global is None:
    #         self._best_input_global = self.min_input
    #     return self._best_input_global

    @property
    def best_input_global_guess(self):
        return self.min_input

    def _f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.repeated_shift
        a, b, c, r, s, t = self.parameters
        output = self.pkg.zeros(x.shape[:-1], **self.array_options)
        for i in range(self.num_repeats):
            output += a * (x[..., 2 * i + 1] - b * x[..., 2 * i] ** 2 + c * x[..., 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[..., 2 * i]) + s + 5 * x[..., 2 * i]
        output /= self.num_repeats
        return output

    def f(self, inputs):
        return (self._f(inputs) - self.best_shift_global) * self.rescale


class QBranin(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim % 2 == 0)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.num_repeats = dim // 2
        self.shift = self.as_array([2.5, 7.5])
        self.repeated_shift = self.shift.repeat(self.num_repeats)
        self.scale = 7.5
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        min_input = self.array([-1/3, -0.2])
        self.min_input = min_input.repeat(1, self.num_repeats)
        self.best_shift_global = 0
        self.target_rescale = 100.
        self.rescale = 1
        self.best_shift_global = self._f(self.best_input_global).min().item()
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global_guess(self):
        return self.min_input

    def _f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.repeated_shift
        a, b, c, r, s, t = self.parameters
        output = self.pkg.zeros(x.shape[:-1], **self.array_options)
        for i in range(self.num_repeats):
            output += a * (x[..., 2 * i + 1] - b * x[..., 2 * i] ** 2 + c * x[..., 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[..., 2 * i]) + s + 5 * x[..., 2 * i] ** 2
        output /= self.num_repeats
        return output

    def f(self, inputs):
        return (self._f(inputs) - self.best_shift_global) * self.rescale

class SQBranin(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim % 2 == 0)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.num_repeats = dim // 2
        self.shift = self.as_array([2.5, 7.5]) + 2.
        self.repeated_shift = self.shift.repeat(self.num_repeats)
        self.scale = 7.5
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        min_input = self.array([-0.6, -0.467])
        self.min_input = min_input.repeat(1, self.num_repeats)
        self.best_shift_global = 0
        self.target_rescale = 100.
        self.rescale = 1
        self.best_shift_global = self._f(self.best_input_global).min().item()
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global_guess(self):
        return self.min_input

    def _f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.repeated_shift
        a, b, c, r, s, t = self.parameters
        output = self.pkg.zeros(x.shape[:-1], **self.array_options)
        for i in range(self.num_repeats):
            output += a * (x[..., 2 * i + 1] - b * x[..., 2 * i] ** 2 + c * x[..., 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[..., 2 * i]) + s + 5 * x[..., 2 * i] ** 2
        output /= self.num_repeats
        return output

    def f(self, inputs):
        return (self._f(inputs) - self.best_shift_global) * self.rescale

class SSQBranin(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim % 2 == 0)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.num_repeats = dim // 2
        self.shift = self.as_array([2.5, 7.5]) + 4.
        self.repeated_shift = self.shift.repeat(self.num_repeats)
        self.scale = 7.5
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        min_input = self.array([-0.867, -0.733])
        self.min_input = min_input.repeat(1, self.num_repeats)
        self.best_shift_global = 0
        self.target_rescale = 100.
        self.rescale = 1
        self.best_shift_global = self._f(self.best_input_global).min().item()
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global_guess(self):
        return self.min_input

    def _f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.repeated_shift
        a, b, c, r, s, t = self.parameters
        output = self.pkg.zeros(x.shape[:-1], **self.array_options)
        for i in range(self.num_repeats):
            output += a * (x[..., 2 * i + 1] - b * x[..., 2 * i] ** 2 + c * x[..., 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[..., 2 * i]) + s + 5 * x[..., 2 * i] ** 2
        output /= self.num_repeats
        return output

    def f(self, inputs):
        return (self._f(inputs) - self.best_shift_global) * self.rescale


class S3QBranin(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim % 2 == 0)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.num_repeats = dim // 2
        self.shift = self.as_array([2.5, 7.5]) + 3.
        self.repeated_shift = self.shift.repeat(self.num_repeats)
        self.scale = 7.5
        self.parameters = (1, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6, 10, 1.0 / (8 * math.pi))
        min_input = self.array([-0.733, -0.6])
        self.min_input = min_input.repeat(1, self.num_repeats)
        self.best_shift_global = 0
        self.target_rescale = 100.
        self.rescale = 1
        self.best_shift_global = self._f(self.best_input_global).min().item()
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global_guess(self):
        return self.min_input

    def _f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        x = inputs * self.scale + self.repeated_shift
        a, b, c, r, s, t = self.parameters
        output = self.pkg.zeros(x.shape[:-1], **self.array_options)
        for i in range(self.num_repeats):
            output += a * (x[..., 2 * i + 1] - b * x[..., 2 * i] ** 2 + c * x[..., 2 * i] - r) ** 2 + \
                      s * (1 - t) * torch.cos(x[..., 2 * i]) + s + 5 * x[..., 2 * i] ** 2
        output /= self.num_repeats
        return output

    def f(self, inputs):
        return (self._f(inputs) - self.best_shift_global) * self.rescale


# one global minimum at [-0.58070678, ...], value 0
# there other local modes: 2^d - 1 @ 0.54936 (value 36)
# value 100 @ 0
# fully additive
class StyblinskiTang(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.scale = 5
        self.min_input = self.as_array(self.pkg.ones(self.dim)) * -2.903534 / self.scale
        self.min_input = self.min_input.reshape(-1, self.dim)
        self.rescale = 1
        self.best_shift_global = 0
        self.best_shift_global = self.f(self.best_input_global).min().item()
        self.target_rescale = 100.
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global_guess(self):
        return self.min_input

    @property
    def worst_input_global(self):
        if self._worst_input_global is None:
            self._worst_input_global = self.as_array(self.pkg.ones(self.dim))
        return self._worst_input_global

    def f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        assert(inputs.shape[-1] == self.dim)
        x = inputs * self.scale
        # modify to mean so that best value does not depend on d
        output = 1/2 * (x ** 4 - 16 * x ** 2 + 5 * x).mean(dim=-1)
        output = (output - self.best_shift_global) * self.rescale
        return output


class Levy(TorchTestFn):
    def __init__(self, dim, *args, min_input=1, **kwargs):
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self.scale = 10
        self.min_input = self.as_array(self.pkg.ones(self.dim)) * self.as_array(min_input) / self.scale
        self.min_input = self.min_input.reshape(-1, self.dim)
        self.rescale = 1
        self.best_shift_global = 0
        self.best_shift_global = self.f(self.best_input_global).min().item()
        self.target_rescale = 100.
        self.rescale = self.target_rescale / self.f(self.as_array(self.pkg.zeros(self.dim))).item()

    @property
    def best_input_global(self):
        if self._best_input_global is None:
            self._best_input_global = self.min_input
        return self._best_input_global

    def f(self, inputs):
        inputs = torch.atleast_2d(inputs)
        assert(inputs.shape[-1] == self.dim)
        x = inputs * self.scale
        min_input = self.min_input.reshape(-1) * self.scale
        w = (x - min_input) / 4.0 + 1.0
        output = torch.sin(math.pi * w[:, 0]) ** 2.0
        for i in range(self.dim - 1):
            output += (w[:, i] - 1) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w[:, i] + 1.0) ** 2.0)
        output += ((w[:, -1] - 1) ** 2 * (1.0 + torch.sin(2 * math.pi * w[:, -1]) ** 2.0))
        output = (output - self.best_shift_global) * self.rescale
        return output


class S35Levy(Levy):
    def __init__(self, dim, *args, min_input=3.5, **kwargs):
        super().__init__(dim, *args, min_input=min_input, **kwargs)


class S5Levy(Levy):
    def __init__(self, dim, *args, min_input=5, **kwargs):
        super().__init__(dim, *args, min_input=min_input, **kwargs)

class S65Levy(Levy):
    def __init__(self, dim, *args, min_input=6.5, **kwargs):
        super().__init__(dim, *args, min_input=min_input, **kwargs)


class Rover60(TorchTestFn):
    def __init__(self, dim, *args, **kwargs):
        assert (dim == 60)
        TorchTestFn.__init__(self, dim, *args, **kwargs)
        self._rover = rover()

    @property
    def best_input_global(self):
        return self.as_array(60 * [np.nan]) # dont know

    @property
    def best_target_global(self):
        return 0

    @property
    def worst_input_global(self):
        return self.as_array(60 * [np.nan])

    @property
    def worst_target_global(self):
        return np.nan

    @property
    def offset(self):
        return self._rover.fn_instance.offset

    def f(self, inputs):
        inputs = (torch.atleast_2d(inputs) + 1) / 2 # from [-1, 1]^60 to [0, 1]^60
        # make function non-negative
        return (-self.as_array([self._rover(xi) for xi in inputs.numpy()]) + self.offset).reshape(-1)


class SRover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = srover()


class CXY10MS0Rover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = mrover(xycost=10)


class MS0Rover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = mrover()


class SGMS0Rover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = mrover(new_goal=np.array([0.8, 0.7]))


class SB1SGMS0Rover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = mrover(new_goal=np.array([0.8, 0.7]), new_start=np.array([0.4, 0.3]))


class SB2SGMS0Rover60(Rover60):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rover = mrover(new_goal=np.array([0.8, 0.7]), new_start=np.array([0.3, 0.5]))


# Log Transformations

class Log10Transform:
    def __init__(self, *args, **kwargs):
        self._in_init = True
        super().__init__(*args, **kwargs)
        self._in_init = False
        self.log_shift = 1e-6

    def f(self, inputs):
        if self._in_init:
            return super().f(inputs)
        outputs = super().f(inputs)
        outputs = torch.log10(outputs + self.log_shift)
        return outputs


class LogRosenbrock(Log10Transform, Rosenbrock):
    pass

class LogSRosenbrock(Log10Transform, SRosenbrock):
    pass

class LogSSRosenbrock(Log10Transform, SSRosenbrock):
    pass

class LogS35Rosenbrock(Log10Transform, S35Rosenbrock):
    pass

class LogS65Rosenbrock(Log10Transform, S65Rosenbrock):
    pass

class LogBranin(Log10Transform, Branin):
    pass

class LogQBranin(Log10Transform, QBranin):
    pass

class LogSQBranin(Log10Transform, SQBranin):
    pass

class LogSSQBranin(Log10Transform, SSQBranin):
    pass

class LogS3QBranin(Log10Transform, S3QBranin):
    pass

class LogStyblinskiTang(Log10Transform, StyblinskiTang):
    pass

class LogLevy(Log10Transform, Levy):
    pass

class LogS35Levy(Log10Transform, S35Levy):
    pass

class LogS5Levy(Log10Transform, S5Levy):
    pass

class LogS65Levy(Log10Transform, S65Levy):
    pass

TORCH_TEST_FN_CHOICES = {'sphere': Sphere, 'gk': GaussianKernels,
                         'rosenb': Rosenbrock, 'branin': Branin, 'styb': StyblinskiTang,
                         'srosenb': SRosenbrock, 'ssrosenb': SSRosenbrock,
                         's35rosenb': S35Rosenbrock, 's65rosenb': S65Rosenbrock,
                         'qbranin': QBranin, 'sqbranin': SQBranin, 'ssqbranin': SSQBranin, 's3qbranin': S3QBranin,
                         'levy': Levy, 's35levy': S35Levy, 's5levy': S5Levy, 's65levy': S65Levy,
                         'rover60': Rover60, 'srover60': SRover60, 'ms0rover60': MS0Rover60,
                         'cxy10ms0rover60': CXY10MS0Rover60, 'sgms0rover60': SGMS0Rover60,
                         'sb1sgms0rover60': SB1SGMS0Rover60, 'sb2sgms0rover60': SB2SGMS0Rover60,
                         'logrosenb': LogRosenbrock, 'logbranin': LogBranin, 'logstyb': LogStyblinskiTang,
                         'logsrosenb': LogSRosenbrock, 'logssrosenb': LogSSRosenbrock,
                         'logs35rosenb': LogS35Rosenbrock, 'logs65rosenb': LogS65Rosenbrock,
                         'logqbranin': LogQBranin, 'logsqbranin': LogSQBranin, 'logssqbranin': LogSSQBranin,
                         'logs3qbranin': LogS3QBranin,
                         'loglevy': LogLevy, 'logs35levy': LogS35Levy, 'logs5levy': LogS5Levy, 'logs65levy': LogS65Levy
                         }

if __name__ == '__main__':
    dim = 20
    seed = 117
    noise = 0
    min_input = 0.42 * np.ones(dim)
    min_bound, max_bound = -1, 1
    max_radius = (max_bound - min_bound) / 2 * float(dim) ** 0.5
    lower_bound = np.full(dim, min_bound)
    upper_bound = np.full(dim, max_bound)
    bounds = np.vstack((lower_bound, upper_bound))
    test = Branin(dim=dim, seed=seed, noise=noise, bounds=bounds)
    print("============ GUESS ===========")
    print("INPUT=", test.best_input_global_guess)
    print()
    print("===========  BEST  ============")
    print("INPUT=", test.best_input_global, "TARGET=", test.best_target_global)
    print()
    print("===========  WORST ============")
    print("INPUT=", test.worst_input_global, "TARGET=", test.worst_target_global)
    print(test.worst_input_global.shape)