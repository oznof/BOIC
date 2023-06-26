from time import time as time
from typing import List

from pulse.caster import Caster
from .data import Data
from .jobs import Job
from .settings import RunSettings, Settings


class Manager(Caster):
    BASE_CLS_RUN_SETTINGS = RunSettings
    BASE_CLS_EXP_SETTINGS = Settings
    BASE_CLS_EXP_DATA = Data

    def __init__(self, cls_run_settings=None, cls_exp_settings=None, cls_exp_data=None, args: List = None,
                 initialize_run=True):
        # input args are a list just like those from sys.argv (if None)
        # after parsing/loading it becomes a dict, stored in run_settings for global settings
        # and then (exp) settings
        if cls_run_settings is None:
            cls_run_settings = self.BASE_CLS_RUN_SETTINGS
        if cls_exp_settings is None:
            cls_exp_settings = self.BASE_CLS_EXP_SETTINGS
        if cls_exp_data is None:
            cls_exp_data = self.BASE_CLS_EXP_DATA
        self.raise_error_if(not issubclass(cls_run_settings, self.BASE_CLS_RUN_SETTINGS))
        self.raise_error_if(not issubclass(cls_exp_settings, self.BASE_CLS_EXP_SETTINGS))
        self.raise_error_if(not issubclass(cls_exp_data, self.BASE_CLS_EXP_DATA))
        self.cls_run_settings = cls_run_settings
        self.cls_exp_settings = cls_exp_settings
        self.cls_exp_data = cls_exp_data
        if initialize_run:
            self.run_settings = cls_run_settings()
            self.run_settings.argload(args=args)
        self.time_start = None
        self.exp_settings = None
        self.exp_data = None
        self.exp_job = None
        self.exp_test = None
        self.exp_model = None

    @property
    def settings(self):
        return self.exp_settings

    @settings.setter
    def settings(self, value):
        self.exp_settings = value

    @property
    def verbose(self):
        return self.run_settings.verbose or self.debug

    @property
    def debug(self):
        return self.run_settings.get('debug', False) or (self.settings and self.settings.get('debug', False))

    @property
    def rng_initialize(self):
        return self.settings.rng_initialize

    @property
    def ninit(self):
        return self.settings.acq_ninit

    @property
    def data(self):
        return self.exp_data

    @property
    def draw_initial_samples(self):
        return self.settings.draw_initial_samples

    @property
    def model(self):
        return self.exp_model

    @model.setter
    def model(self, value):
        self.exp_model = value

    @property
    def model_cls(self):
        return self.settings.model_cls

    @property
    def model_options(self):
        return self.settings.model_options

    @property
    def train_options(self):
        return self.settings.train_options

    @property
    def model_train_inputs(self):
        return self.data.model_train_inputs

    @property
    def model_train_targets(self):
        return self.data.model_train_targets

    @property
    def model_last_known_state(self):
        return self.data.model_states[-1] if self.data.model_states else None

    @property
    def test(self):
        return self.exp_test

    @test.setter
    def test(self, value):
        self.exp_test = value

    @property
    def test_cls(self):
        return self.settings.eval_cls

    @property
    def test_options(self):
        return self.settings.eval_options

    def get_time(self, start=None):
        if start is None:
            start = self.time_start
        return time() - start

    def exp_run(self, args):
        if self.exp_start(args):
            try:
                while not self.data.is_complete:
                    self.exp_step()
            except Exception as e:
                raise e
            finally:
                self.exp_end()

    def exp_start(self, args=None):
        self.time_start = time()
        self.exp_settings = self.cls_exp_settings()
        self.exp_settings.packages_initialize()  # initialize package options if needed (e.g. torch default tensor)
        self.exp_settings.argload(args=args)  # at this point all parsing and loading should be done
        self.exp_job = Job(path=self.exp_settings.path, name=self.exp_settings.name, overwrite=self.debug)
        start = self.exp_job.start
        if start:
            self.exp_model = None
            self.exp_data = Data(self)
            self.test_load()
            self.show_exp_initialization()
        return start

    def test_load(self):
        test_cls = self.str_to_module_obj(self.test_cls)
        self.test = test_cls(**self.test_options)
        if self.data.n and self.test.noise > 0:  # generate noise to sync internal rng
            self.test.generate_noise(self.data.n)

    def exp_step(self):
        self.model_acquire()
        self.model_train()
        self.data.process()  # stats['state'] has self.model.state(), used in exp_load

    def model_acquire(self):
        time_start = time()
        is_initial = not self.data.has_data
        self.rng_initialize(self.data.seed)
        train_inputs = self.draw_initial_samples() if is_initial else self.model.acquire(**self.settings.acq_options)
        self.data.update(train_inputs)
        if is_initial:
            self.model_load(n=0)
            if self.debug:
                try:
                    print(self.data.train_targets)
                    print(self.model.train_targets)
                    print(self.model.train_targets.mean())
                    print(self.model.mean_module.constant.item())
                    print("HAS LOADED MODEL")
                except:
                    pass
        self.data.time_acq.append(self.get_time(time_start))
        # TODO: info_acq ?

    def model_load(self, n=-1, verbose=True):
        self.settings.load_model_options(train_inputs=self.data.model_train_inputs,
                                         train_targets=self.data.model_train_targets)
        self.model = self.settings.model_cls(**self.settings.model_options)
        if self.data.has_data:
            n_data = self.data.ninit + n if n >= 0 else n
            self.model.set_train_data(self.model_train_inputs[:n_data], self.model_train_targets[:n_data],
                                      opt_data=self.data)
        if self.model_last_known_state:
            if self.debug and verbose:
                self.print_sep()
                print("PRIOR MODEL STATE: ", self.model.state(), '\n')
            self.model.state(state=self.model_last_known_state if n == -1 else self.data.model_states[n])
            if self.debug and verbose:
                print("AFTER LOADING: ", self.model.state())
        return self.model

    def model_train(self):
        time_start = time()
        self.rng_initialize(self.data.seed)
        train_info = self.model.train(**self.train_options)
        if self.data.save_train:
            self.data.info_train.append(train_info)
        if self.data.save_model_state:
            self.data.model_states.append(self.model.state(copy=True, to_numpy=True))
        self.data.time_train.append(self.get_time(time_start))

    def dump(self):
        exp = {'cls_run_settings': self.module_obj_to_str(self.cls_run_settings),
               'cls_exp_settings': self.module_obj_to_str(self.cls_exp_settings),
               'cls_exp_data': self.module_obj_to_str(self.cls_exp_data),
               'run_settings': self.run_settings.dump(),
               'exp_settings': self.exp_settings.dump(),
               'exp_data': self.exp_data.dump()
               }
        return exp

    def exp_end(self):
        if self.data.is_complete:
            self.exp_job.end(self.dump())

    @classmethod
    def exp_load(cls, file: str, load_model=True, run_to_completion=True, verbose=True):
        exp_job = Job(file=file)
        exp = exp_job.load()
        cls_run_settings = exp.get('cls_run_settings')
        cls_exp_settings = exp.get('cls_exp_settings')
        cls_exp_data = exp.get('cls_exp_data')
        cls_run_settings = cls.str_to_module_obj(cls_run_settings)
        cls_exp_settings = cls.str_to_module_obj(cls_exp_settings)
        cls_exp_data = cls.str_to_module_obj(cls_exp_data)
        run_settings = exp.get('run_settings')
        new = cls(cls_run_settings, cls_exp_settings, cls_exp_data, initialize_run=run_settings is None)
        new.exp_job = exp_job
        new.exp_settings = new.cls_exp_settings.load(exp['exp_settings'])
        if run_settings:
            new.run_settings = new.cls_run_settings.load(run_settings)
        else:
            new.run_settings.repeats = 1
            new.run_settings._counter = 1
            new.run_settings._has_loaded = True
            new.run_settings.global_args = None
            new.run_settings.seed = new.exp_settings.seed
            new.run_settings.debug = new.exp_settings.get('debug', False)
            new.run_settings.verbose = new.exp_settings.get('verbose', True)
        new.exp_settings.packages_initialize()
        new.exp_data = new.cls_exp_data.load(exp['exp_data'], manager=new)
        new.test_load()
        if load_model or run_to_completion:
            new.model_load()
        if verbose:
            new.show_exp_loading()
        if run_to_completion and not new.data.is_complete:
            new.time_start = time()
            while not new.data.is_complete:
                new.exp_step()
            new.exp_end()
        return new

    def show_exp_initialization(self):
        print(f'Path: {self.settings.path}')
        print(f'Name: {self.settings.name}')
        if self.verbose:
            self.show_exp_initialization_extra()

    def show_exp_initialization_extra(self):
        pass

    def show_exp_loading(self):
        print(f'Path: {self.settings.path}')
        print(f'Name: {self.settings.name}')
        if self.verbose:
            self.show_exp_loading_extra()

    def show_exp_loading_extra(self):
        pass

    def run(self):
        while not self.run_settings.is_complete:
            exp_args = self.run_settings.next_exp_args
            self.exp_run(exp_args)

    @classmethod
    def print_sep(cls, sep='=', times=100):
        print(sep * times, '\n')