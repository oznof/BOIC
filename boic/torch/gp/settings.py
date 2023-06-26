import os
from copy import copy

from boic.torch.settings import TorchSettings
from pulse.caster import try_str_to_bool_or_number


class GPTorchSettings(TorchSettings):
    DESCRIPTION = 'Bayesian Optimization with GP'
    DEFAULT_PATH = os.path.join(TorchSettings.DEFAULT_PATH, 'gp')

    DEFAULT_TRAIN_METHOD = 'lbfgsb'
    TRAIN_METHOD_CHOICES = ['adam', 'lbfgsb']
    @classmethod
    def train_method_condition(cls, value):
        return value in cls.TRAIN_METHOD_CHOICES
    DEFAULT_TRAIN_METHOD_MODE = 'base'
    TRAIN_METHOD_MODE_CHOICES = ['base']
    @classmethod
    def train_method_mode_condition(cls, value):
        return value in cls.TRAIN_METHOD_MODE_CHOICES
    @property
    def default_train_mode(self):
        return super().default_train_mode + self.sd + f'M{self.default_train_method},MM{self.default_train_method_mode}'
    TRAIN_MODE_DICT = copy(TorchSettings.TRAIN_MODE_DICT)
    TRAIN_MODE_DICT.update({'method': ('M', str, 'train_method_condition'),
                            'method_mode': ('MM', str, 'train_method_mode_condition')
                          })

    # the default acquisition function is expected improvement
    DEFAULT_ACQ_FN = 'ei'
    ACQ_FN_CHOICES = ['ei', 'trgreedyei']
    @classmethod
    def acq_fn_condition(cls, value):
        return value in cls.ACQ_FN_CHOICES
    DEFAULT_ACQ_METHOD = 'lbfgsb'
    ACQ_METHOD_CHOICES = ['adam', 'lbfgsb']
    @classmethod
    def acq_method_condition(cls, value):
        return value in cls.ACQ_METHOD_CHOICES
    DEFAULT_ACQ_METHOD_MODE = 'base'
    ACQ_METHOD_MODE_CHOICES = ['base']
    @classmethod
    def acq_method_mode_condition(cls, value):
        return value in cls.ACQ_METHOD_MODE_CHOICES
    @property
    def default_acq_mode(self):
        return super().default_acq_mode + self.sd + f'F{self.default_acq_fn},M{self.default_acq_method},' \
                                                    f'MM{self.default_acq_method_mode}'
    ACQ_MODE_DICT = copy(TorchSettings.ACQ_MODE_DICT)
    ACQ_MODE_DICT.update({'fn': ('F', str, 'acq_fn_condition'),
                          'method': ('M', str, 'acq_method_condition'),
                          'method_mode': ('MM', str, 'acq_method_mode_condition')
                          })

    # the baseline gp model should have a constant mean and ard stationary matern kernel
    DEFAULT_MODEL_MEAN_MODE = 'const'
    MODEL_MEAN_MODE_CHOICES = ['const', 'quad']
    @classmethod
    def model_mean_mode_condition(cls, value):
        return value in cls.MODEL_MEAN_MODE_CHOICES
    DEFAULT_MODEL_BASE_KERNEL = 'matern'
    MODEL_BASE_KERNEL_CHOICES = ['matern', 'rbf']
    @classmethod
    def model_base_kernel_condition(cls, value):
        return value in cls.MODEL_BASE_KERNEL_CHOICES
    DEFAULT_MODEL_BASE_KERNEL_MODE = 'ard'
    MODEL_BASE_KERNEL_MODE_CHOICES = ['ard', 'shared']
    @classmethod
    def model_base_kernel_mode_condition(cls, value):
        return value in cls.MODEL_BASE_KERNEL_MODE_CHOICES
    DEFAULT_MODEL_KERNEL = 'sk'
    # stationary, cylindrical and informative kernels
    MODEL_KERNEL_CHOICES = ['sk', 'ck', 'ik']
    @classmethod
    def model_kernel_condition(cls, value):
        return value in cls.MODEL_KERNEL_CHOICES
    DEFAULT_MODEL_KERNEL_MODE = 'base'
    MODEL_KERNEL_MODE_CHOICES = ['base', 'fixed', 'greedy']
    @classmethod
    def model_kernel_mode_condition(cls, value):
        return value in cls.MODEL_KERNEL_MODE_CHOICES

    @property
    def default_model_mode(self):
        return f'MM{self.default_model_mean_mode},BK{self.default_model_base_kernel},BKM{self.default_model_base_kernel_mode},' \
               f'K{self.default_model_kernel},KM{self.default_model_kernel_mode}'
    MODEL_MODE_DICT = {'mean_mode': ('MM', str, 'model_mean_mode_condition'),
                       'base_kernel': ('BK', str, 'model_base_kernel_condition'),
                       'base_kernel_mode': ('BKM', str, 'model_base_kernel_mode_condition'),
                       'kernel': ('K', str, 'model_kernel_condition'),
                       'kernel_mode': ('KM', str, 'model_kernel_mode_condition')}

    @property
    def require_commands_extra(self):
        flag = False
        self['model_mode'] = self.args['model_mode']
        if self['model']['kernel'] == 'ik' and (
           any([self['model']['kernel_mode'].startswith(kmode)
                for kmode in ['fixed', 'greedy']])):
            flag = True
        return flag

    def get_commands_extra(self):
        # at this point, require_ecommands_extra has already been run
        if self['model']['kernel'] == 'ik' and (
            any([self['model']['kernel_mode'].startswith(kmode)
                for kmode in ['fixed', 'greedy']])):
            # bool is not supported in argparse by default (unless using store_true)
            commands = {'pool': dict(type=self.str_to_bool, dest='model_pool', choices=[True, False],
                                     default='true')}
            return commands, self.get_commands_extra_ik_fixed
        else:
            raise ValueError

    def get_commands_extra_ik_fixed(self):
        if self.args['model_pool']:
            commands = dict(nsl=dict(type=try_str_to_bool_or_number, dest='model_nsl', required=True),
                            nsr=dict(type=try_str_to_bool_or_number, dest='model_nsr', required=True))
            getter = self.get_commands_extra_ik_fixed_pool
        else:
            commands = dict(nslv=dict(type=try_str_to_bool_or_number, dest='model_nslv'),
                            nsrv=dict(type=try_str_to_bool_or_number, dest='model_nsrv'),
                            nsll=dict(type=try_str_to_bool_or_number, dest='model_nsll'),
                            nsrl=dict(type=try_str_to_bool_or_number, dest='model_nsrl'))
            getter = self.get_commands_extra_ik_fixed_not_pool
        return commands, getter

    def get_commands_extra_ik_fixed_pool(self):
        a = self.args
        a['model_nslv'] = a['model_nsll'] = a['model_nsl']
        a['model_nsrv'] = a['model_nsrl'] = a['model_nsr']
        a['model_has_nsv'] = True
        a['model_has_nsl'] = True
        return {}, None  # stop

    def get_commands_extra_ik_fixed_not_pool(self):
        a = self.args
        if a['model_nslv'] is not None and a['model_nsrv'] is not None:
            a['model_has_nsv'] = True
        else:
            a['model_has_nsv'] = False
        if a['model_nsll'] is not None and a['model_nsrl'] is not None:
            a['model_has_nsl'] = True
        else:
            a['model_has_nsl'] = False
        if not a['model_has_nsv'] and not a['model_has_nsl']:
            raise ValueError('If ik_fixed_not_pool, must provide (1) -nslv -nsrv or/and (2) -nsll -nsrl')
        return {}, None  # stop

    @property
    def excluded_keys(self):
        return super().excluded_keys + ('model_has_nsv', 'model_has_nsl')

    @property
    def default_name(self):
        name = super().default_name
        if self['model']['kernel'] == 'ik' and (
           any([self['model']['kernel_mode'].startswith(kmode)
                for kmode in ['fixed', 'greedy']])):
            name, extras = name.split(' pool:')
            has_cls = 'cls' in self['model']
            if has_cls:
                cls = self['model'].pop('cls')
            extras = self.ik_fixed_extras(**self['model'])
            if has_cls:
                self['model']['cls'] = cls
            name = f'{name} {extras}'
        return name

    @classmethod
    def ik_fixed_extras(cls, **d):
        pool = d['pool']
        extras = f'P{cls.bool_to_str(pool)}'
        if pool:
            nsl = d['nsl']
            nsr = d['nsr']
            return extras + f',L{nsl},R{nsr}'
        nslv = d.get('nslv')
        nsrv = d.get('nsrv')
        if nslv is not None and nsrv is not None:
            extras += f',LV{nslv},RV{nsrv}'
        nsll = d.get('nsll')
        nsrl = d.get('nsrl')
        if nsll is not None and nsrl is not None:
            extras += f',LL{nsll},RL{nsrl}'
        return extras

if __name__ == '__main__':
    f = GPTorchSettings()
    f.packages_initialize()
    f.argload()
    print(f)
    print("="*30)
    print(f.name)