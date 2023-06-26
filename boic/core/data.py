from datetime import timedelta
import numpy as np

from pulse.errors import raise_error_if
from pulse.containers import AttributeFieldDict

from .models import SurrogateModel
from .settings import RunSettings, Settings
from .tests import TestFn


class Data(AttributeFieldDict):
    ## STRUCTURE
    DEFAULT_FIELDS = ('train', 'info', 'model', 'time', 'stats')

    def __init__(self, manager=None):
        # object.__setattr__(self, '_in_init', True)
        self._manager = manager
        if self.manager:
            raise_error_if(not self.manager.exp_settings.has_loaded,
                           error=RuntimeError, msg='Exp settings must load first.')
        super().__init__()
        # train data
        self.train_inputs = None
        self.train_targets = None
        # other info related to train or acquisition, considered relevant for post-processing
        self.info_train = []
        self.info_acq = []
        # model states (parameters mostly, so that models can be load given initial settings)
        self.model_states = []
        # computation time
        self.time_train = []
        self.time_acq = []
        self.time_n = []
        self.time_total = 0
        # NOTE: All of these can be computed once an experiment ends given train data
        # stats related to current acquisitions
        self.stats_curr_input = []
        self.stats_curr_input_gap_mae = []
        self.stats_curr_input_gap_linf =[]
        self.stats_curr_input_best_mae = []
        self.stats_curr_input_best_linf = []
        self.stats_curr_input_acq_mae = []
        self.stats_curr_input_acq_linf = []
        self.stats_curr_target = []
        self.stats_curr_target_opt_gap = []
        self.stats_curr_target_unit_gap = []
        # stats related to best acquisitions
        self.stats_best_input = []
        self.stats_best_input_gap_mae = []
        self.stats_best_input_gap_linf = []
        self.stats_best_target_init = None
        self.stats_best_target = []
        self.stats_best_target_opt_gap = []
        self.stats_best_target_unit_gap = []
        self.stats_best_ai = None  # best acquisition improvement (over initial conditions)
        self.stats_best_nai = None # normalized acquisition improvement (equivalent to target_unit_gap, but only last)
        # depreciation indicators
        self.stats_best_awi = []  # number of acquisitions without improvement over best (incumbent), nmax + 1
        self.stats_best_nawi = [] # normalized version
        self.stats_best_ntwi = None  # normalized total without improvement (measures constant learning)
        # depreciated improvements
        self.stats_best_ai_twi = None
        self.stats_best_nai_t_ntwi = None  # normalized indicator considering improvement size and constant learning
        self.stats_best_hm_nai_ntwi = None
        self.stats_best_mdi = None
        self.stats_best_nmdi = None
        #
        self.stats_best_auc = [] # area under curve
        self.stats_best_a_1_2_5 = {}  # acquisitions to reach 1/1, 1/2, 1/5, 1/10, ... 1/100
        # global
        self.stats_best_input_global = None
        self.stats_best_target_global = None
        self.stats_worst_input_global = None
        self.stats_worst_target_global = None

    @classmethod
    def load(cls, d, manager=None):
        new = super().load(d)
        new._manager = manager
        return new

    @property
    def manager(self):
        return self._manager

    @property
    def time_start(self):
        return self.manager.time_start

    @property
    def run_settings(self) -> RunSettings:
        return self.manager.run_settings

    @property
    def settings(self) -> Settings:
        return self.manager.settings

    @property
    def options(self):
        return self.settings.stats_options

    @property
    def debug(self):
        return self.manager.debug or self.options.get('debug', False)

    @property
    def verbose(self):
        return self.manager.verbose

    @property
    def as_array(self):
        return self.settings.as_array

    @property
    def to_numpy(self):
        return self.settings.to_numpy

    @property
    def dim(self):
        return self.settings.dim

    @property
    def ninit(self):
        return self.settings.acq_ninit

    @property
    def nmax(self):
        return self.settings.acq_nmax

    @property
    def has_data(self):
        return self.train_targets is not None

    @property
    def n(self):
        return (len(self.train_targets) - self.ninit) if self.has_data else 0

    @property
    def seed(self):
        return self.settings.seed + self.n

    @property
    def is_complete(self):
        return self.n == self.nmax

    @property
    def minimize(self):
        return  self.settings.get('acq_minimize', True)

    @property
    def standardize(self):
        return self.settings.standardize

    @property
    def logarithmic(self):
        return self.settings.logarithmic

    @property
    def best(self):
        return np.min if self.minimize else np.max

    @property
    def argbest(self):
        return np.argmin if self.minimize else np.argmax

    def diff(self, x, y):
        return x - y if self.minimize else y - x

    @property
    def model(self) -> SurrogateModel:
        return self.manager.model

    @property
    def test(self) -> TestFn:
        return self.manager.test

    @property
    def best_target_init(self):
        if self.has_data and self['stats']['best_target_init'] is None:
            self['stats']['best_target_init'] = self.best(self.train_targets[:self.ninit])
        return self['stats']['best_target_init']

    @property
    def best_input_global(self) -> [None, np.array] :
        if self['stats']['best_input_global'] is None and self.test.best_input_global is not None:
            self['stats']['best_input_global'] = self.test.to_numpy(self.test.best_input_global)
        return self['stats']['best_input_global']

    @property
    def best_target_global(self) -> [None, float]:
        if self['stats']['best_target_global'] is None and self.test.best_target_global is not None:
            self['stats']['best_target_global'] = self.test.to_numpy(self.test.best_target_global)
        return self['stats']['best_target_global']

    @property
    def worst_input_global(self) -> [None, np.array] :
        if self['stats']['worst_input_global'] is None and self.test.worst_input_global is not None:
            self['stats']['worst_input_global'] = self.test.to_numpy(self.test.worst_input_global)
        return self['stats']['worst_input_global']

    @property
    def worst_target_global(self) -> [None, float]:
        if self['stats']['worst_target_global'] is None and self.test.worst_target_global is not None:
            self['stats']['worst_target_global'] = self.test.to_numpy(self.test.worst_target_global)
        return self['stats']['worst_target_global']

    @property
    def stats_save(self):
        return self.options.get('save', True)

    @property
    def save_model(self):
        return self.options.get('save_model', True)

    @property
    def save_model_it(self):
        return self.options.get('save_model_it', 1)

    @property
    def save_model_state(self):
        return self.stats_save and self.save_model and self.n % self.save_model_it == 0

    @property
    def save_train(self):
        return self.options.get('save_train', True)

    @property
    def throw_errors(self):
        return self.options.get('throw_errors', True)

    @property
    def model_train_inputs(self):
        return None if self.train_inputs is None else self.as_array(self.train_inputs)

    @model_train_inputs.setter
    def model_train_inputs(self, value):
        train_targets = self.test.eval(value)
        value = self.to_numpy(value)
        self.train_inputs = value if self.train_inputs is None else np.vstack((self.train_inputs, value))
        self.model_train_targets = train_targets

    @property
    def model_train_targets(self):
        if self.train_targets is None:
            return None
        targets = self.train_targets
        if self.logarithmic:
            targets = np.log(targets + np.exp(-12))
        if self.standardize:
            mean = targets.mean()
            std = targets.std()
            targets = (targets - mean) / (std if std >= 1e-9 else 1)
        return self.as_array(targets)

    @model_train_targets.setter
    def model_train_targets(self, value):
        value = self.to_numpy(value)
        self.train_targets = value if self.train_targets is None else np.hstack((self.train_targets, value))
        if self.model is not None:
            self.model.set_train_data(self.model_train_inputs, self.model_train_targets,
                                      opt_data=self)

    def update(self, train_inputs):
        # NOTE: this only updates the training data in the model if it exists,
        # in some cases it may be useful to generate a model only after observing some data
        # for this reason, initial train data is first processed here and only then
        # model settings are loaded, leading to the creation of a model (see model_create)
        self.model_train_inputs = train_inputs # everything else follows from the setters

    def process(self):
        self.time_n.append(self.time_acq[-1] + self.time_train[-1])
        self.time_total += self.time_n[-1]
        best_id =  self.argbest(self.train_targets)
        curr_input =  self.train_inputs[-1]
        best_input =  self.train_inputs[best_id]
        if self.best_input_global is None:
            curr_input_gap_mae = best_input_gap_mae = None
            curr_input_gap_linf = best_input_gap_linf = None
        else:
            curr_input_gap = np.abs(curr_input - self.best_input_global)
            best_input_gap = np.abs(best_input - self.best_input_global)
            curr_input_gap_mae = np.min(np.mean(curr_input_gap, axis=-1))
            best_input_gap_mae = np.min(np.mean(best_input_gap, axis=-1))
            curr_input_gap_linf = np.min(np.max(curr_input_gap, axis=-1))
            best_input_gap_linf = np.min(np.max(best_input_gap, axis=-1))
        cb_input_gap = np.abs(curr_input - best_input)
        curr_input_best_mae = np.mean(cb_input_gap)
        curr_input_best_linf = np.max(cb_input_gap)
        if self.n <= 1:
            curr_input_acq_mae = curr_input_acq_linf = 0.
        else:
            cacq_input_gap = np.abs(curr_input - self.train_inputs[self.ninit:-1])
            curr_input_acq_mae = np.min(np.mean(cacq_input_gap, axis=-1))
            curr_input_acq_linf = np.min(np.max(cacq_input_gap, axis=-1))
        curr_target = self.train_targets[-1]
        best_target = self.train_targets[best_id]
        if self.best_target_global is None:
            curr_target_opt_gap = best_target_opt_gap = None
            curr_target_unit_gap = best_target_unit_gap = None
        else:
            curr_target_opt_gap = self.diff(curr_target, self.best_target_global)
            best_target_opt_gap = self.diff(best_target, self.best_target_global)
            init_target_opt_gap = self.diff(self.best_target_init, self.best_target_global)
            # can be negative, indicating that curr_target is worse than the best init sample
            # in which case it quantifies the factor of how much worse it is compared to the init opt gap
            curr_target_unit_gap = self.diff(self.best_target_init, curr_target) / init_target_opt_gap
            # always between 0 and 1
            best_target_unit_gap = self.diff(self.best_target_init, best_target) / init_target_opt_gap
        locald = locals()
        locald.pop('self')
        if self.stats_save:
            stats = self['stats']
            stats_keys = stats.keys()
            for k, v in locald.items():
                if k in stats_keys:
                    stats[k].append(v)
        if self.verbose:
            self.show(**locald)

    def show(self, **kwargs):
        best_id = kwargs.get('best_id')
        best_input = kwargs.get('best_input')
        best_input_gap_mae = kwargs.get('best_input_gap_mae')
        best_input_gap_linf = kwargs.get('best_input_gap_linf')
        best_target = kwargs.get('best_target')
        curr_input = kwargs.get('curr_input')
        curr_input_gap_mae = kwargs.get('curr_input_gap_mae')
        curr_input_gap_linf = kwargs.get('curr_input_gap_linf')
        curr_target = kwargs.get('curr_target')
        best_id = max(best_id + 1 - self.ninit, 0)
        if best_input_gap_mae is None:
            best_input_gap_mae = np.inf
        if best_input_gap_linf is None:
            best_input_gap_linf = np.inf
        if curr_input_gap_mae is None:
            curr_input_gap_mae = np.inf
        if curr_input_gap_linf is None:
            curr_input_gap_linf = np.inf

        if self.dim <= 10:
            print(f'N={self.n} BEST=(#:{best_id}, I:{best_input}, MAE:{best_input_gap_mae:.3e}, T:{best_target:.3e})  '
                  f'CURR=(I:{curr_input}, MAE:{curr_input_gap_mae:.3e}, T:{curr_target:.3e})  '
                  f'TIME=(N:{timedelta(seconds=self.time_n[-1])}, TOT:{timedelta(seconds=self.time_total)}).')
        else:
            print(f'N={self.n} BEST=(#:{best_id}, LINF:{best_input_gap_linf:.3e},'
                  f' MAE:{best_input_gap_mae:.3e} T:{best_target:.3e})  '
                  f'CURR=(LINF:{curr_input_gap_linf:.3e}, MAE:{curr_input_gap_mae:.3e}, T:{curr_target:.3e})  '
                  f'TIME=(N:{timedelta(seconds=self.time_n[-1])}, TOT:{timedelta(seconds=self.time_total)}).')
        if self.debug:
            print("TRAIN TIME:", timedelta(seconds=self.time_train[-1]),
                  " ACQ TIME:", timedelta(seconds=self.time_acq[-1]))

    def dump(self):
        data = {}
        if self.stats_save:
            # compute/load these
            _ = self.worst_input_global # see getters for these two
            _ = self.worst_target_global
            best_targets = self.to_numpy(self.stats_best_target)
            self.stats_best_ai = self.target_gap(best_targets)
            self.stats_best_nai = self.nai(best_targets, self.best_target_global, only_last=True)
            self.stats_best_awi = list(self.awi(best_targets))
            self.stats_best_nawi = list(self.nawi(best_targets))
            self.stats_best_ntwi = self.ntwi(best_targets)
            self.stats_best_ai_twi = self.ai_twi(best_targets)
            self.stats_best_nai_t_ntwi = self.nai_t_ntwi(best_targets, self.best_target_global)
            self.stats_best_hm_nai_ntwi = self.hm_nai_ntwi(best_targets, self.best_target_global)
            self.stats_best_mdi = self.mdi(best_targets)
            self.stats_best_nmdi = self.nmdi(best_targets, self.best_target_global)
            self.stats_best_auc = list(self.auc(best_targets, self.best_target_global))
            self.stats_best_a_1_2_5 = self.a_1_2_5(best_targets)
            data = super().dump()
        else:
            data['train_inputs'] = self.train_inputs
            data['train_targets'] = self.train_targets
        return data

    @classmethod
    def best_targets(cls, targets: np.ndarray, ninit=1, minimize=True):
        best = np.min if minimize else np.max
        best_targets = np.zeros(len(targets) - ninit + 1)
        for i in range(ninit, len(targets) + 1):
            best_targets[i-ninit] = best(targets[:i])
        return best_targets

    @classmethod
    def target_gap(cls, targets: np.ndarray, to_best=False, ninit=1, minimize=True):
        # measures acquisition improvement (over initial conditions)
        targets = cls.best_targets(targets, ninit=ninit, minimize=minimize) if to_best else targets
        diff = (lambda x, y: x - y) if minimize else (lambda x, y: y - x)
        return diff(targets[0], targets[-1])

    @classmethod
    def nai(cls, targets: np.ndarray, best_target_global: float, only_last=False, to_best=False, ninit=1,
            minimize=True):
        if best_target_global is None:
            return None
        # same as unitary gap/ normalized target_gap if best_target_global is known
        targets = cls.best_targets(targets, ninit=ninit, minimize=minimize) if to_best else targets
        diff = (lambda x, y: x - y) if minimize else (lambda x, y: y - x)
        target_init = targets[0]
        if only_last:
            targets = targets[-1]
        nai = diff(target_init, targets) / diff(target_init, best_target_global)
        return nai

    @classmethod
    def awi(cls, targets: np.ndarray, is_best=True, remove_initial=False, **kwargs):
        # counter: acquisitions without improvement
        # this only makes sense if given targets are best, if not, make sure to set is_best=False
        best_targets = targets if is_best else cls.best_targets(targets, **kwargs)
        awi = np.zeros_like(best_targets)
        iai = np.abs(np.diff(best_targets))  # instantaneous acquisition improvement
        awi[1] = 1
        for i in range(2, len(awi)):
            awi[i] = 1 if iai[i-2] != 0 else awi[i-1] + 1
        if remove_initial:
            awi = awi[1:]
        return awi

    @classmethod
    def nawi(cls, targets: np.ndarray, is_best=True, remove_initial=True, **kwargs):
        awi = cls.awi(targets, is_best=is_best, remove_initial=remove_initial, **kwargs)
        len_awi = len(awi) if remove_initial else len(awi) - 1  # don't consider initial
        return awi / len_awi

    @classmethod
    def ai_twi(cls, targets: np.ndarray, is_best=True, **kwargs):
        ai = cls.target_gap(targets, to_best=not is_best, **kwargs) # this quantifies acquisition improvement
        # ignore first entry, initial conditions, as it does not contain proper acquisitions, referring to seed points
        awi = cls.awi(targets, is_best=is_best, remove_initial=True, **kwargs)
        # mai = ai / len(awi)      # mean acquisition improvement
        twi = np.sum(awi)      # total (acquisitions/counts) without improvement
        return ai / twi
    # NOTE: the factor len(awi), which is nmax (number of proper acquisitions), cancels out when computing
    # mai / mean awi. The denominator in ai / sum(awi) can also be written as sum of differences
    # sum(diff(best_targets)) / sum(awi), which is the sum of best improvement differences over
    # sum of acquisition (counts) without improvement (total acquisitions/count without improvement).

    # For two optimizers that have same best value from same initial conditions, ai is going to be the same, not
    # taking into account the optimization trajectory. However, an optimizer that constantly improves has a
    # lower mean/total awi, so it would be better according to this indicator.
    # EXAMPLE:
    # assume nmax = 5, and the best of initial is 30 (that is, both start at 30)
    # >>> best_targets_1 = [30., 20., 20., 20., 10., 10.]
    # >>> awi_1 =          [0.,  1.,  1.,  2.,  3.,  1.]
    # >>> ai_1 = 20.
    # >>> twi_1 = 8.
    # >>> ai_twi_1 = 2.5
    # >>> best_targets_2 = [30., 20., 20., 15., 10., 10.]
    # >>> awi_2 =          [0.,  1.,  1.,  2.,  1.,  1.]
    # >>> ai_2 = 20.
    # >>> twi_2 = 6.
    # >>> ai_twi_2 = 3.(3)
    # both final values are the same and in fact the trajectory is somewhat similar,
    # but there is a slight advantage for method 2 because it is able to better improve over the incumbent,
    # so this indicator assigns a greater value to methods that constantly improve.
    # These indicators can then be normalized to a reference of constant improvement,
    # which is simply the mean improvement
    # >>> ai_3 = 20.
    # >>> awi_3 =          [0.,  1.,  1.,  1.,  1.,  1.]
    # >>> ai_twi_3 = 4.
    # normalized ai_twi_1 = 0.625, normalized ai_twi_2 = 0.834.
    # this is equivalent to the normalized twi, which does not take into account the size of
    # improvement as it cancels.
    @classmethod
    def ntwi(cls, targets: np.ndarray, is_best=True, **kwargs):
        awi = cls.awi(targets, is_best=is_best, remove_initial=True, **kwargs)
        twi = np.sum(awi)
        return len(awi) / twi

    @classmethod
    def nai_t_ntwi(cls, targets: np.ndarray, best_target_global: float, to_best=False, ninit=1, minimize=True):
        if best_target_global is None:
            return None
        targets = cls.best_targets(targets, ninit=ninit, minimize=minimize) if to_best else targets
        nai = cls.nai(targets, best_target_global, only_last=True)
        ntwi = cls.ntwi(targets, is_best=True)
        return nai * ntwi

    @classmethod
    def hm_nai_ntwi(cls, targets: np.ndarray, best_target_global: float, to_best=False, ninit=1, minimize=True):
        # harmonic mean
        if best_target_global is None:
            return None
        targets = cls.best_targets(targets, ninit=ninit, minimize=minimize) if to_best else targets
        nai = cls.nai(targets, best_target_global, only_last=True)
        ntwi = cls.ntwi(targets, is_best=True)
        return 2 / (1/nai + 1/ntwi)

    # In turn, while not the same,  instantaneous_improv_j / count_without_improv_j
    # gives the depreciated instantaneous improvement at j. So that delayed improvements are valued less.
    # The sum of depreciated instantaneous improvements is the (total) depreciated improvement. By taking the mean,
    # we are computing the mean depreciated improvement.
    # For comparison,
    # >>> ai_twi_1 = 2.5
    # >>> ai_twi_2 = 3.(3)
    # >>> mdi_1 = 2.(6)
    # >>> mdi_2 = 3.5
    # >>> iai_1             = [10.,  0.,  0., 10.,  0.]
    # >>> awi_1[1:]         = [1.,   1.,  2.,  3.,  1.]
    # >>> iai_1 / awi_2[1:] = [10.,  0.,  0., 3.3,  0.]
    # >>> iai_2             = [10.,  0.,  5.,  5.,  0.]
    # >>> awi_2[1:]         = [1.,   1.,  2.,  1.,  1.]
    # >>> iai_2 / awi_2[1:] = [10.,  0., 2.5,  5.,  0.]

    @classmethod
    def mdi(cls, targets: np.ndarray, is_best=True, ninit=1, minimize=True):
        best_targets = targets if is_best else cls.best_targets(targets, ninit=ninit, minimize=minimize)
        iai = np.abs(np.diff(best_targets))  # instantaneous acquisition improvement
        awi = cls.awi(best_targets, remove_initial=True)
        return np.mean(iai / awi)

    @classmethod
    def nmdi(cls, targets: np.ndarray, best_target_global: float, is_best=True, ninit=1, minimize=True):
        # normalized version
        if best_target_global is None:
            return None
        best_targets = targets if is_best else cls.best_targets(targets, ninit=ninit, minimize=minimize)
        nai = cls.nai(best_targets, best_target_global, only_last=False, to_best= False)
        inai = np.abs(np.diff(nai))  # instantaneous normalized acquisition improvement
        awi = cls.awi(best_targets, remove_initial=True)
        return np.mean(inai / awi)

    @classmethod
    def auc(cls, targets: np.ndarray, best_target_global: float, is_best=True, ninit=1, minimize=True):
        # area under best trajectory curve
        if best_target_global is None:
            return None
        best_targets = targets if is_best else cls.best_targets(targets, ninit=ninit, minimize=minimize)
        nai = cls.nai(best_targets, best_target_global, only_last=False, to_best=False)
        # initial conditions are 0 / 1 and we want to have division by nmax instead of nmax+1
        # exclude initial condition as acquisition from count
        auc = np.array([np.sum(nai[:i + 1]) / max(i, 1) for i in range(0, len(nai)-1)])
        return auc

    @classmethod
    def a_1_n(cls, targets: np.ndarray,  n: int = 2, is_best=True, ninit=1, minimize=True):
        # (number of) acquisitions to 1/n
        best_targets = targets if is_best else cls.best_targets(targets, ninit=ninit, minimize=minimize)
        beq = (lambda x, y: x <= y) if minimize else (lambda x, y: x >= y)
        tinit = best_targets[0]
        t = np.argmax(beq(best_targets, tinit / n))
        if t == 0 and n > 1: # did not reach
            t = None
        return t

    @classmethod
    def a_1_2_5(cls, targets: np.ndarray, is_best=True, ninit=1, minimize=True):
        return {n: cls.a_1_n(targets, n, is_best, ninit, minimize) for n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]}