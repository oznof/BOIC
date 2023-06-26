import math
import sys

import numpy as np
from scipy.stats.qmc import Sobol

from pulse.containers import ParsingDict
from .tests import TestFn, TEST_FN_CHOICES

# NOTE: The convention for parse dicts is to use upper case prefix identifiers and lower case for values
# if they are strings or booleans, e.g. defining a boolean as 'false' by default will then turn it into proper boolean
# False. Using booleans directly as default values also works, but the strings generated would not be consistent
# as some would have the first letter in uppercase and others in lowercase, despite both leading to the same setting.
# This might also lead to parsing bugs involving the definition of UNIQUE prefixes, e.g.
# prefix 'FOOT' and prefix 'FOO' with value True that is converted to str -> same substring 'FOOT'.

class RunSettings(ParsingDict):
    DEFAULT_FIELDS = ('run',)
    IMMUTABLE_AFTER_LOADING = False

    # default class values for arguments
    DEFAULT_SEED = 117
    DEFAULT_REPEATS = 1
    DEFAULT_DEBUG = 'false'
    DEFAULT_VERBOSE = 'true'


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    @property
    def default_commands(self):
        d = self.field_delimiter
        commands = [(f'seed', dict(type=int, default=self.default_seed,
                                   help='Run base seed from which experiment seeds are generated.')),
                    (f'repeats', dict(type=int, default=self.default_repeats,
                                      help='Number of repeats using different seeds.')),
                    (f'debug', dict(type=self.str_to_bool, default=self.default_debug)),
                    (f'verbose', dict(type=self.str_to_bool, default=self.default_verbose))]
        return dict(commands)

    def argload(self, args=None, **kwargs):
        # this parses only initial args, others are supposed to be passed on to exp
        # in particular, it gets the seed, and repeats
        # relevant args, gets the additional arguments (extras) which will feed
        # to exp settings.
        # NOTE: ignore kwargs
        exp_args = super().argload(args=args, parse_all=False, return_extras=True)[1]
        self['global_args'] = exp_args # NOTE: it does not contain seed as these

    @property
    def next_exp_args(self):
        self.raise_error_if(not self.has_loaded)
        if self.is_complete:
            sys.exit(1)
        exp_seed = self.seed + self.counter
        self._counter += 1
        return self['global_args'] + ['-seed', str(exp_seed)]

    @property
    def is_complete(self):
        return self.has_loaded and self.counter == self.repeats

    def dump(self):
        d = super().dump()
        d.update({'_counter': self._counter})
        return d


class Settings(ParsingDict):
    ########################################################################################################
    ##### PLATFORM DEPENDENT CLASSES AND METHODS
    # NOTE: other packages need to override these, e.g. torch, tensorflow, jax, ...
    PKG = np
    RNG_CLS = np.random.RandomState

    @property
    def pkg(self):
        return self.PKG

    @property
    def rng_cls(self):
        return self.RNG_CLS

    @property
    def array_cls(self):
        return np.ndarray

    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def as_array(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    @classmethod
    def to_numpy(cls, x):
        return np.array(x)

    def packages_initialize(self):
        np.set_printoptions(precision=3)
        # ...

    def rng_initialize(self, seed=None):
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        return seed

    ##########################################################################################################
    ### STRUCTURE
    DEFAULT_FIELDS = ('exp', 'eval', 'prior', 'train', 'acq', 'model', 'stats')
    DEFAULT_FIELD_DELIMITER = '_'  # Field delimiter, after which a subfield is assumed
    SPECIAL_SUBFIELD = 'mode'
    DEFAULT_SUBFIELD_DELIMITER = ','  # delimiter for parsing special strings such as those given in special subfields
    DEFAULT_ATTR_DELIMITER = '_'
    IMMUTABLE_AFTER_LOADING = True
    ### DEFAULT DUMPS (NAME GENERATION)
    DUMPS_WITH_ALL_REFERENCE_KEYS = False  # override any other behavior and display all reference keys (argparsed dict)
    EXCLUDE_EMPTY_FIELDS = True
    EXCLUDE_EMPTY_VALUES = True
    EXCLUDED_SUBFIELDS_IF_DEFAULT_VALUE = True

    @property
    def excluded_keys(self):
        return ('path', 'name')

    @property
    def excluded_fields(self):
        return ('stats')

    @property
    def excluded_subfields(self):
        return ()

    def excluded_subfields_if(self, key, value):
        # if certain subfields are to be excluded, do so based on whether they have the same default values as
        # those given initially in commands
        return super().excluded_subfields_if(key, value)

    @property
    def custom_name(self): # override custom name getter
        # modify custom name to include seed if not in name
        cname = self.args['name']
        if 'seed' not in cname:
            cname = f'exp{self.fd} seed:{self.seed}' + self.args['name']
        return cname

    # DEFAULT_NAME_WITH_DEFAULT_SPECIAL_SUBFIELDS = True  # include in name special subfields even default values
    # DEFAULT_NAME_WITH_EMPTY_VALUES = False # None or empty strings, dicts


    ##########################################################################################################
    ### EXP SETTINGS
    DEFAULT_PATH = '~/storage/data/boic/'
    DEFAULT_SEED = 117

    @property
    def eval_seed(self): # add a reference to eval too
        return self['exp']['seed']
    # by inheritance, other existing aliases are self.seed, self['seed'], self.exp_seed
    ##########################################################################################################
    ### EVAL SETTINGS
    # MAKE SURE TO CHANGE THESE TWO IN SUBCLASSES  !!!
    EVAL_FN_CLS = TestFn  # this is the class of the test functions
    EVAL_FN_DICT = TEST_FN_CHOICES  # eval_name should be one of the keys, allowing the conversion to class
    EVAL_FN_ARGS = EVAL_FN_CLS.ARGS # ('dim', 'seed', 'noise', 'bounds')
    @classmethod
    def eval_name_condition(cls, name):
        return name in cls.EVAL_FN_DICT
    @property
    def default_eval_name(self):
        # print(self.eval_fn_dict.keys())
        # print(list(self.eval_fn_dict.keys())[0])
        return list(self.eval_fn_dict.keys())[0]
    DEFAULT_EVAL_DIM = 1
    DEFAULT_EVAL_NOISE = 0  # observation noise variance
    DEFAULT_EVAL_BOUNDS_TYPE = 'ccube'  # centered cube [-1, 1]^dim
    DEFAULT_EVAL_DEVICE_TYPE = 'cpu'
    # DEFAULT_EVAL_MODE
    @property
    def default_eval_mode(self):
        return f'E{self.default_eval_name},D{self.default_eval_dim},N{self.default_eval_noise},' \
               f'B{self.default_eval_bounds_type},P{self.pkg.__name__},V{self.default_eval_device_type}'
    # parser dicts of the form {field}_{subfield}_dict.upper() =
    # {subfield1 : (prefix1, cast fn, condition[optional], (value gen, args)[optional]),  ...}
    # The third entry is optional and is a condition to be evaluated after applying cast fn to get a value,
    # forcing value to obey condition
    # If the third entry is given, a fourth entry can also be specified so that in case condition fails, a
    # value generator function is run, using that value instead (value is evaluated once to see if its valid, otherwise
    # throws ValueError during loading). this is specified as a tuple (gen, required args), the required args are a list
    # of names that are supposed to fetch from instance. Functions can also be specified as string in which case
    # it calls getattr
    # the following is set in self as ['eval']['dim'] ['eval']['noise'] (same as ['eval_dim'], ['eval_noise'])
    EVAL_MODE_DICT = {'name': ('E', str, 'eval_name_condition'),
                      'dim': ('D', int),
                      'noise': ('N', ParsingDict.str_to_number),
                      'bounds_type': ('B', str),
                      'pkg': ('P', str),
                      'device_type': ('V', str)}

    @property
    def eval_bounds_type(self):
        return self['eval']['bounds_type']

    @eval_bounds_type.setter
    def eval_bounds_type(self, value):
        self.raise_error_if(value != 'ccube', f'{value} is invalid for acq_bounds_type.')
        dim = self.dim
        min_bound, max_bound = -1, 1
        max_radius = (max_bound - min_bound) / 2 * float(dim) ** 0.5
        lower_bound = np.full(dim, min_bound)
        upper_bound = np.full(dim, max_bound)
        bounds = np.vstack((lower_bound, upper_bound))
        self['eval']['bounds'] = bounds
        self['eval']['max_radius'] = max_radius

    # custom setter to be run at the end of argload to generate an updated string, cls and options
    @property
    def eval_mode(self):  # this definition should not be required, but need to define setter: if values have changed
        return self['eval']['mode']  # that affect the string, need to update it last, for this reason, modes should
                                      # always be defined last in command list (argparse)
    @eval_mode.setter
    def eval_mode(self, value):
        # ignore value because, others have already been loaded, fetch them directly (mode has been loaded too)
        ## self['eval']['mode'] = f'E{self.eval_name},D{self.eval_dim},N{self.eval_noise},'
        ## self['eval']['mode'] += f'B{self.eval_bounds_type},P{self.pkg},V(self.eval_device_type)'
        # get class from name, this is required as well as eval_options in manager
        self['eval']['cls'] = self.eval_fn_dict[self.eval_name]
        self['eval']['options'] = {k: getattr(self, f'eval{self.ad}{k}') for k in self.eval_fn_args}

    @property
    def dim(self): #  set up another alias for self.eval_dim
        return self['eval']['dim']

    ##########################################################################################################
    ### PRIOR SETTINGS
    DEFAULT_PRIOR_INPUT = 'center'
    PRIOR_INPUT_CHOICES = ['center', 'best', 'worst']
    @classmethod
    def prior_input_condition(cls, value):
        return value in cls.PRIOR_INPUT_CHOICES

    @property
    def default_prior_mode(self):
        return f'X{self.default_prior_input}'

    PRIOR_MODE_DICT = {'input_type': ('X', str, 'prior_input_condition')}

    @property
    def prior_mode(self):
        return self['prior']['mode']

    @prior_mode.setter
    def prior_mode(self, value):
        if self['prior']['input_type'] == 'center':
            self['prior']['input'] = np.mean(self.eval_bounds, axis=0)
        elif self['prior']['input_type'] == 'best':
            test = self['eval']['cls'](**self['eval']['options'])
            self['prior']['input'] = None if test.best_input_global is None else test.to_numpy(test.best_input_global)
            self['prior']['input'] = np.atleast_2d(self['prior']['input'])[0] # only take first if many
        elif self['prior']['input_type'] == 'worst':
            test = self['eval']['cls'](**self['eval']['options'])
            self['prior']['input'] = None if test.worst_input_global is None else test.to_numpy(test.worst_input_global)
            self['prior']['input'] = np.atleast_2d(self['prior']['input'])[0] # only take first if many
        else:
            raise NotImplementedError(f"prior.input_type={self['prior']['input_type']} not implemented")


    ##########################################################################################################
    ### TRAIN SETTINGS
    # target (output) mapping/warping
    DEFAULT_TRAIN_TMAP = 'id'
    TRAIN_TMAP_CHOICES = ['id', 'log', 'std']
    @classmethod
    def train_tmap_condition(cls, value):
        return value in cls.TRAIN_TMAP_CHOICES
    @property
    def default_train_mode(self):
        return  f'TM{self.default_train_tmap}'
    TRAIN_MODE_DICT = {'tmap': ('TM', str, 'train_tmap_condition')}

    @property
    def logarithmic(self):
        return self['train']['tmap'] == 'log'

    @property
    def standardize(self):
        return self['train']['tmap'] == 'std'

    @property
    def train_options(self):
        try:
            return self['train']['options']
        except KeyError:
            return {}
    ##########################################################################################################
    ### ACQ SETTINGS
    DEFAULT_ACQ_MINIMIZE = 'true'
    DEFAULT_ACQ_BOUNDS_TYPE = 'eval'  # same as eval
    @property
    def acq_bounds_type(self):
        return self['acq']['bounds']

    @acq_bounds_type.setter
    def acq_bounds_type(self, value):
        self.raise_error_if(value != 'eval', f'{value} is invalid for acq_bounds_type.')
        # dim = self.dim
        # min_bound, max_bound = -1, 1
        # max_radius = (max_bound - min_bound) / 2 * float(dim) ** 0.5
        # lower_bound = np.full(dim, min_bound)
        # upper_bound = np.full(dim, max_bound)
        # bounds = np.vstack((lower_bound, upper_bound))
        self['acq']['bounds'] = self['eval']['bounds']
        self['acq']['max_radius'] = self['eval']['max_radius']
    DEFAULT_ACQ_NINIT = -1  # if negative, adaptive based on dim: runs function below (during argload)
    DEFAULT_ACQ_SCRAMBLE_INIT = 'true'
    @classmethod
    def default_acq_ninit_fn(cls, dim):
        return  2 ** min(math.ceil(np.log2(dim + 1)), 4)
    DEFAULT_ACQ_NMAX = 200

    @property
    def default_acq_mode(self):
        return f'B{self.default_acq_bounds_type},NI{self.default_acq_ninit},' \
               f'SI{self.default_acq_scramble_init},NM{self.default_acq_nmax}'

    ACQ_MODE_DICT = {#'minimize': ('PMIN', Settings.str_to_bool),
                     'bounds_type': ('B', str),
                     # NOTE: it depends on eval_dim, but because the args are executed in order, eval_dim has been
                     # loaded already
                     'ninit': ('NI', int, lambda v: v >= 0, ('default_acq_ninit_fn', ['dim'])),
                     'scramble_init': ('SI', ParsingDict.str_to_bool),
                     'nmax':  ('NM', int)}

    @property
    def acq_minimize(self):
        return self.str_to_bool(self.DEFAULT_ACQ_MINIMIZE)

    @property
    def acq_options(self):
        try:
            return self['acq']['options']
        except KeyError:
            return {}

    def draw_initial_samples(self):
        samples =  self.draw_sobol_samples(bounds=self.to_numpy(self.acq_bounds),
                                           initial_samples=self.acq_ninit, seed=self.seed,
                                           scramble=self.acq_scramble_init)
        if self['prior']['input_type'] != 'best' and self['prior']['input'] is not None:
            prior_input = self['prior']['input']
            samples = np.vstack((prior_input, samples))
            samples = samples[np.sort(np.unique(samples, axis=0, return_index=True)[1])][:self.acq_ninit]
        return self.as_array(samples)

    @classmethod
    def draw_sobol_samples(cls, bounds, initial_samples, seed=None, scramble=False):
        dim = bounds.shape[-1]
        raw_samples = Sobol(dim, scramble=scramble, seed=seed).random(initial_samples)
        samples = bounds[0] + (bounds[1] - bounds[0]) * raw_samples
        return samples

    ##########################################################################################################
    ### MODEL SETTINGS
    DEFAULT_MODEL_MODE = ''
    # TODO:

    ##########################################################################################################
    ### STATS SETTINGS
    # don't change these for now as saving model states is required for loading experiments later on if needed
    # save at every iteration
    DEFAULT_STATS_SAVE = 'true'
    DEFAULT_STATS_SAVE_MODEL = 'true'
    DEFAULT_STATS_SAVE_MODEL_IT = 1
    DEFAULT_STATS_SAVE_TRAIN = 'true'
    DEFAULT_STATS_THROW_ERRORS = 'true'

    DEFAULT_STATS_MODE = f'S{DEFAULT_STATS_SAVE},SM{DEFAULT_STATS_SAVE_MODEL},SMIT{DEFAULT_STATS_SAVE_MODEL_IT},' + \
                         f'ST{DEFAULT_STATS_SAVE_TRAIN},TE{DEFAULT_STATS_THROW_ERRORS}'
    STATS_MODE_DICT = {'save': ('S', ParsingDict.str_to_bool),
                       'save_model': ('SM', ParsingDict.str_to_bool),
                       'save_model_it': ('SMIT', int),
                       'save_train': ('ST', ParsingDict.str_to_bool),
                       'throw_errors': ('TE', ParsingDict.str_to_bool)}
    # this is required for stats module (linked via manager)
    @property
    def stats_options(self):
        return self['stats']

    ##########################################################################################################
    ##########################################################################################################
    ### ARGPARSE COMMANDS
    @property
    def default_commands(self):
        # the parsed dict from commands is stored in self.args, this will always remain unchanged
        # command history is stored in self.commands
        # argument names should be constructed such that the first substring is a field up to
        # self.field_delimiter
        # if not given, it will add the argument to 'run' field so that the command name 'path'
        # without run{field_delimiter}path still goes to settings[run][path], so by default
        # settings[run] is populated, unless a substring in arg name matches a different specified
        # field name
        # in addition to arg name, a parse dict to argparse must be given
        d = self.fd
        # NOTE: the order of the commands is relevant in the creation of the default global name
        commands = [(f'path', dict(type=str, default=self.default_path)),
                    # if the name is not provided, see name:
                    # it is going to be the dict of args to str, excluding fields
                    # in get_name_excluded_arg_fields if DEFAULT_NAME_WITH_ALL_ARGS = False
                    # name can be accessed as self.name which is stored in self[run][name]
                    # the original name is in self.args['name'] which is None if not given (no custom name)
                    (f'name', dict(type=str, help='Results filename.')),
                    (f'seed', dict(type=int, default=self.default_seed, help='Exp seed.')),
                    (f'eval{d}mode', dict(type=str, default=self.default_eval_mode)),
                    (f'prior{d}mode', dict(type=str, default=self.default_prior_mode)),
                    (f'train{d}mode', dict(type=str, default=self.default_train_mode)),
                    (f'acq{d}mode', dict(type=str, default=self.default_acq_mode)),
                    (f'model{d}mode', dict(type=str, default=self.default_model_mode)),
                    (f'stats{d}mode', dict(type=str, default=self.default_stats_mode))]
        commands = dict(commands)
        return commands