import itertools
import math

import numpy as np
import scipy.optimize


class TestFn:
    RNG_CLS = np.random.RandomState
    PKG = np
    ARGS = ('dim', 'seed', 'noise', 'bounds') # seed, noise and bounds are optional
    MINIMIZE = True
    PRUNE_VERTICES_N_DIM = 20
    DIVERSIFY_N_GUESSES = 5
    MAX_N_GUESSES = 500 # for high dim, it can take too long on opt for finding default input best/worst global

    def __init__(self, dim, seed=None, noise=None, bounds=None, rng_state=None, **kwargs):
        self.dim = dim
        self.seed = seed
        self.noise = None if seed is None else noise # without seed don't generate noise
        self.rng = None
        if self.noise:
            self.rng_initialize(state=rng_state)
        self.bounds = bounds  # bounds are only required for determining a worst possible value
        self._best_input_global = None
        self._best_target_global = None
        self._worst_input_global = None
        self._worst_target_global = None

    @property
    def pkg(self):
        return self.PKG

    @property
    def rng_cls(self):
        return self.RNG_CLS

    @property
    def array_cls(self):
        return np.ndarray

    def array(self, x, *args, **kwargs):
        return np.array(x, *args, **kwargs)

    def as_array(self, x, *args, **kwargs):
        return np.asarray(x)

    @property
    def array_options(self):
        return {}

    @classmethod
    def to_numpy(cls, x, *args, **kwargs):
        return np.array(x, *args, **kwargs)

    @property
    def minimize(self):
        return self.MINIMIZE

    @classmethod
    def better(cls, x, y):
        return  x < y if cls.MINIMIZE else x > y

    @classmethod
    def best(cls, x):
        return  cls.PKG.min(x) if cls.MINIMIZE else cls.PKG.max(x)

    @classmethod
    def worse(cls, x, y):
        return  not cls.better(x, y)

    @classmethod
    def worst(cls, x):
        return cls.PKG.max(x) if cls.MINIMIZE else cls.PKG.min(x)

    @classmethod
    def argbest(cls, x):
        return  cls.PKG.argmin(x) if cls.MINIMIZE else cls.PKG.argmax(x)

    @classmethod
    def argworst(cls, x):
        return cls.PKG.argmax(x) if cls.MINIMIZE else cls.PKG.argmin(x)
    # instance because as_array may depend on instance too
    def argbetter(self, x):
        return self.PKG.argsort(x) if self.minimize else self.as_array(self.to_numpy(self.PKG.argsort(x))[::-1])

    def argworse(self, x):
        return self.as_array(self.argbetter(x)[::-1])

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = None if value is None else self.as_array(value)

    @property
    def diversify_n_guesses(self):
        return self.DIVERSIFY_N_GUESSES

    @property
    def max_n_guesses(self):
        return self.MAX_N_GUESSES

    @property
    def prune_vertices_n_dim(self):
        return self.PRUNE_VERTICES_N_DIM

    @property
    def best_input_global_guess(self):   # this is used in scipy optimize to determine a possible best global if the best is not known
        return None                      # with sufficient precision, it can be multiple guesses (vstacked)

    @property
    def best_input_global(self):
        # NOTE: the default provided here is just some heuristic it won't work in general,
        # make sure it is sensible for a specific test function by providing it directly or
        # a more informative guess
        if self._best_input_global is None:  # if finding some "best" or "worst" involves some computations
                                             # cache the results, so that it does not need to do them every time
                                             # by default try optimizing from some guess
            if self.best_input_global_guess is not None:
                best_guess = self.best_input_global_guess
            elif self.bounds is not None:
                best_guess = self.pkg.mean(self.bounds, 0)   # use mean
            else:  # try origin
                best_guess = self.pkg.zeros(self.dim, **self.array_options)
            # print("BEST GUESS: ", best_guess)
            self._best_input_global = self.pkg.atleast_2d(self.opt(best_guess))
        return self._best_input_global
        # if not provided, some stats are not computed during optimization
        # best_input_global can return multiple inputs in which case during the computation of stats
        # minimum values are taken, that is to find a point that is closer to any of the best

    @property
    def opt(self):
        return self.sciopt

    @property
    def default_sciopt_method(self):
        return 'L-BFGS-B'

    @property
    def default_sciopt_options(self):
        return None

    def sciopt(self, guesses, bounds=None, minimize=None, method=None, jac=None, options=None,
               return_as_array=True, return_res=False):
        minimize = self.minimize if minimize is None else minimize
        # print(minimize)
        try:
            fun = getattr(self, 'f_and_grad')  # define this if one has access to gradients
            # print("OK")
            f =  (lambda x: [self.to_numpy(o) for o in fun(x)]) if minimize else\
                 (lambda x: [-self.to_numpy(o) for o in fun(x)])
            jac = True
            # print("OK")
        except AttributeError:
            f =  lambda x: self.to_numpy(self.f(x)) if minimize else (lambda x: -self.to_numpy(self.f(x)))
        guesses = self.to_numpy(guesses).reshape(-1, self.dim)  # handle multiple guesses
        bounds = self.bounds if bounds is None else bounds
        method = self.default_sciopt_method if method is None else method
        options = self.default_sciopt_options if options is None else options
        if bounds is not None and bounds.shape == (2, self.dim):
            bounds = bounds.T
        res = None
        for x0 in guesses:
            cres = scipy.optimize.minimize(f, x0, method=method, jac=jac, bounds=bounds, options=options)
            if res is None or cres['fun'] < res['fun']:
                res = cres
        if not minimize:
            res['fun'] = - res['fun']
            res['jac'] = - res['jac']
            res['hess_inv'] = - res['hess_inv']
        res['x'] = self.as_array(res['x']) if return_as_array else res['x']
        # print("res x: ", res['x'])
        return res['x'] if not return_res else res

    @property
    def best_target_global(self):
        if self.best_input_global is None:
            return None  # just in case there are multiple global best, all targets however should be the same
        if self._best_target_global is None:
            self._best_target_global = float(self.best(self.f(self.best_input_global)))  # without noise
        return self._best_target_global

    @property
    def worst_input_global(self): # this should depend on the bounds provided, shape (2, dim), so that vstack((lb, ub))
        if self.bounds is None: # worst cases usually happen in one of the vertices of bounding box so
            return None
        if self._worst_input_global is None:
            bounds = self.to_numpy(self.bounds) + 0. # make sure is float array
            # AS DEFAULT TRY VERTICES, BUT CHECK THAT IT IS INDEED THE CASE !!! OVERRIDE THIS IF IT DOES NOT MAKE
            # SENSE FOR A SPECIFIC FUNCTION WHERE IT IS KNOWN NOT TO PERFORM WORSE NEAR THE BOUNDARY
            remaining_dim = self.dim
            start_dim = 0
            cutoff_dim = self.prune_vertices_n_dim # cannot list all vertices if high dim, break them down by
            iw = None                              # assuming independence
            while start_dim < self.dim:
                niw = self.to_numpy([e for e in itertools.product(*bounds[:, start_dim:][:, :cutoff_dim].T)])\
                      + 0.
                if iw is None:
                    iw = niw
                else:
                    liw = len(iw)
                    lniw = len(niw)
                    if lniw < liw:
                        niw = niw.repeat(math.ceil(liw / lniw), axis=0)[:liw]
                    iw = np.hstack((iw, niw))
                start_dim += niw.shape[-1]
            iw = self.as_array(iw)
            # print("IW: ", iw.shape, iw.max(), iw.min())
            # there might be multiple inputs equally worse, argworst only retrieves one of them, use the following instead
            tw = self.f(iw)
            wtw = self.worst(tw)
            wiw = iw[tw == wtw].reshape(-1, self.dim)
            # print(wiw.shape)
            # print()
            # print("wiw: ", wiw)
            lwiw = len(wiw)
            dn = self.diversify_n_guesses - lwiw # diversify guesses up to self.diversify_n_guesses
            if dn > 0 and len(tw) >= self.diversify_n_guesses:
                # print("IN: ", self.argworse(tw))
                wiw = self.pkg.vstack((wiw, iw[self.argworse(tw)[lwiw:][:self.diversify_n_guesses - lwiw]]))
            # print(wiw.shape, wiw.min(), wiw.max())
            # move away a bit from bounds and use these as guess points
            delta = 0.2 / math.sqrt(self.dim) * (self.bounds[1] - self.bounds[0]) / 2 # factor in dim, so that it decreases
            delta = self.as_array(self.to_numpy(delta).repeat(len(wiw), axis=0)).reshape(-1, self.dim)
            guesses = self.as_array(wiw)
            in_lb = wiw == self.bounds[0]
            in_ub = wiw == self.bounds[1]
            if self.pkg.any(in_lb):
                guesses[in_lb] += delta[in_lb]
            if self.pkg.any(in_ub):
                guesses[in_ub] -= delta[in_ub]
            # print("GUESSES: ", guesses[:self.max_n_guesses])
            # print()

            wiwopt = self.opt(guesses[:self.max_n_guesses], minimize=not self.minimize).reshape(-1, self.dim)
            iw = self.pkg.vstack((wiw, wiwopt))
            tw = self.f(iw)
            wtw = self.worst(tw)
            self._worst_input_global = self.as_array(np.unique(np.atleast_2d(self.to_numpy(iw[tw == wtw])), axis=0))
        return self._worst_input_global

    @property
    def worst_target_global(self):
        if self.worst_input_global is None:
            return None
        if self._worst_target_global is None:
            self._worst_target_global = float(self.worst(self.f(self.worst_input_global)))
        return self._worst_target_global


    def rng_initialize(self, seed=None, state=None):
        if seed:
            self.seed = seed
        self.rng = self.RNG_CLS(self.seed)
        if state is not None:
            self.rng.set_state(state)

    @property
    def rng_state(self):
        if isinstance(self.rng, self.RNG_CLS):
            return self.rng.get_state()
        return None

    def f(self, inputs):  # without noise
        raise NotImplementedError

    def eval(self, inputs):
            inputs = np.array(inputs)
            inputs = inputs.reshape(-1, self.dim) if len(inputs.shape) == 1 else inputs
            targets = self.f(inputs)
            if self.noise:
                targets +=  self.generate_noise(targets.shape)
            return targets

    def generate_noise(self, shape, noise=None, rng=None, mode='normal'):
        if noise is None:
            noise = self.noise
        if rng is None:
            rng = self.rng
        return getattr(self, 'generate_noise_' + mode)(shape, noise, rng)

    @staticmethod
    def generate_noise_normal(shape, noise=1, rng=np.random):
        return math.sqrt(noise) * rng.randn(*shape)

    def to_dict(self):
        options = self.__dict__
        if self.noise:
            rng = options.pop('rng')
            options['rng_state'] = rng.get_state()
        return options

    @classmethod
    def from_dict(cls, **options):
        return cls(**options)


class TestDummy(TestFn):

    def __init__(self,*args, min_input=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_input= self.pkg.zeros(self.dim) if min_input is None else self.as_array(min_input)

    def f(self, inputs):
        return np.sum((inputs - self.min_input) ** 2, axis=-1)

TEST_FN_CHOICES = {'test': TestDummy}