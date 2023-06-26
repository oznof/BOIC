from typing import Callable, Dict

import numpy as np
import pandas as pd

from pulse.containers import AttributeFieldDict, ParsingDict
from .jobs import IO
from .settings import Settings
from .managers import Manager

from boic.utils.plotting import plot_df


class Aliaser(IO, AttributeFieldDict):
    BASE_CLS_SETTINGS = Settings
    # DEFAULT_FIELDS = CLS_SETTINGS.DEFAULT_FIELDS  #   ('exp', 'eval', 'prior', 'train', 'acq', 'model', 'stats')
    # SPECIAL_SUBFIELD = 'mode'
    # DEFAULT_EVAL = {'N': {'0': ''},
    #                 'B': {'ccube': ''},
    #                 'P': {'numpy': ''},
    #                 'V': {'cpu': ''}}
    # DEFAULT_PRIOR = {'X': {'center': ''}}
    # DEFAULT_TRAIN = {'TM': {'id': ''}}
    # DEFAULT_ACQ = {'B': {'eval': ''}, 'NI': {'16': ''}, 'SI': {'true': ''}, 'NM': {'200': ''} }
    REVERSAL_PREFIX = '@' # aliaser maps from original to alias, this reverses the map

    ALIASES =  {'acq': 'Acquisition',
                'it': 'Iteration',
                'best_target_unit_gap': 'Normalized Improvement',
                'curr_input_acq_mae': 'Input MAE Previous Acquisitions',
                'curr_input_best_mae': 'Input MAE Incumbent',
                'curr_input_gap_mae': 'Input MAE Optimal Gap',
                'best_auc': 'Improvement Rate'}
    # DEFAULT_OPTIONS = {'eval_starts_with': 'E', 'prior_starts_with': 'X',
    #                    'train_starts_with': 'TM', 'acq_starts_with': 'B'}
    DEFAULT_OPTIONS = {'default_field': 'stats',
                       'field_delimiter': '::',
                       'special_subfield': 'mode',
                       'subfield_delimiter': ',',
                       'extra_delimiter': ',',
                       'remove': [],
                       'remove_if_default': True,
                       'remove_prefix_if_str': False,
                       'remove_value_if_true': True,
                       'remove_prefix_if_false': True,
                       'sep': '::',
                       'remove_field': False
                       }


    def __init__(self, settings: Settings = None, options: Dict = None, aliases: Dict = None,
                 default_field=None):
        if settings is None:
            settings = self.BASE_CLS_SETTINGS()
        if not settings.has_loaded:
            settings.argload(parse_all=False)
        self._settings = settings
        if not default_field:
            default_field = self.DEFAULT_OPTIONS.get('default_field')
        super().__init__(fields=self.list_unique([default_field] + list(settings.fields)),
                         default_field=default_field,
                         field_delimiter=settings.delimiter, attr_delimiter=settings.attr_delimiter)
        self._options = self.DEFAULT_OPTIONS.copy()
        self.options.update(options or {})
        for field in self.fields:
            self[field]['caster'] = {} # from str
            self[field]['parser'] = {} # parsing prefixes and value map
            self[field]['aliaser'] = {}  # forward maps
            self[field]['reverser'] = {} # reverses maps
            self[field]['parsed'] = {} # parsed info for a given alias
        self[self.default_field]['aliaser'].update(self.ALIASES)
        self[self.default_field]['aliaser'].update(aliases or {})
        self.initialize()

    @property
    def settings(self):
        return self._settings

    @property
    def options(self):
        return self._options

    @property
    def special_subfield(self):
        return self.settings.get('special_subfield', self.options['special_subfield'])

    @property
    def ed(self):
        return self.options['extra_delimiter']

    @property
    def rp(self):
        return self.REVERSAL_PREFIX

    @property
    def name(self):
        name = self.settings.__class__.__name__.lower().rstrip('settings')
        return name if name else 'base'

    def initialize(self):
        for field in self.fields[1:]:
            this = self[field]
            remove_if_default = self.get_options(key='remove_if_default', default=False, field=field)
            parse_dict = self.settings.to_parse_dict_name(field + self.settings.ad + self.special_subfield)
            try:
                parse_dict = getattr(self.settings, parse_dict)
            except (AttributeError, KeyError):
                continue
            for k, (prefix, typ, *other) in parse_dict.items():
                value = None
                try:
                    value = getattr(self, f'default{self.ad}{field}'.upper())[prefix]
                    has_value = True
                except (AttributeError, KeyError):
                    has_value = False
                if not has_value:
                    try:
                        value = self.settings[field][k]
                        value = {value: '' if remove_if_default else
                                self.bool_to_str(value) if isinstance(value, bool) else
                                self.module_obj_to_str(value)}
                    except KeyError:
                        continue
                this['aliaser'][k] = prefix
                self.reverse(prefix, k, field=field)
                # if prefix in this['parser']:
                #     this['parser'][prefix].update(value or {})
                # else:
                this['parser'][prefix] = value or {}
                this['caster'][prefix] = typ
            try:
                other_defaults = getattr(self, f'default{self.ad}{field}'.upper())
                for k, v in  other_defaults.items():
                    this[k].update(v)
            except (AttributeError, KeyError):
                pass

    def get_options(self, key, default=None, field=None):
        if field:
            return self.options.get(f'{field}{self.fd}{key}', self.options.get(key, default))
        return self.options.get(key, default)

    def reverse(self, alias, value=None, field=None, **kwargs):
        this = self[field] if field else self
        if value is None:
            return this['reverser'].get(alias)
        this_reverser = this['reverser']
        if alias in this_reverser:
            this_reverser_alias = this['reverser'][alias]
            if isinstance(this_reverser_alias, list):
                if value not in this_reverser_alias:
                    this_reverser_alias.append(value)
            elif this_reverser_alias != value:
                    this_reverser[alias] = [this_reverser_alias, value]
        else:
            this_reverser[alias] =  value
        return this_reverser[alias]

    def __call__(self, value, *args, **kwargs):
        if isinstance(value, str):
            values = value.split(self.rp)
            if len(values) == 1:
                for field in self.fields[1:]:
                    if value.startswith(field + self.fd):
                        name = value.replace(field + self.fd, '')
                        alias =  self.field_alias(name, field=field)
                        if alias == name:
                            return self.alias(value, *args, **kwargs)
                return self.alias(value, *args, **kwargs)
            if len(values) == 2:
                value, field = values
                if field == self.name:
                    field = self.default_field # None
                return self.reverse(value, field=field, **kwargs)

    def alias(self, txt, *args, **kwargs):
        name = self['aliaser'].get(txt, self.aliaser(txt))
        if not name:
            name = txt
        return name

    def field_alias(self, txt, field):
        name = self[field]['aliaser'].get(txt, self.field_aliaser(txt, field))
        if not name:
            name = txt
        return name

    def aliaser(self, txt, return_parsed=False):
        # txt = self.file_name(txt)
        alias = ''
        field_sep = ''
        parsed = {}
        skip_fields = self.get_options(key='skip_fields', default=[])
        # print("START ALIASER")
        for field in self.fields[1:]:
            if field in skip_fields:  # default (first) is always skipped
                continue
            # field_prefix = self.get_options(key='', default=)
            # print(field)
            field_alias, field_parsed = self.field_aliaser(txt, field)
            field_sep = self.get_options(key='sep', default='  ', field=field)
            if field_alias:
                alias += field_alias + field_sep
            if field_parsed:
                parsed[field] = field_parsed
        alias = alias.rstrip(field_sep)
        self['aliaser'][txt] = alias
        if parsed:
            self['parsed'][alias] = parsed
        self.reverse(alias, txt, field=self.default_field)
        if not alias:
            alias = txt
        parsed['alias'] = alias
        return (alias, parsed) if return_parsed else alias

    def field_aliaser(self, txt, field):
        if field == self.default_field:
            return self[field].get(txt, '')
        field_txt = self.field_from_str(txt, field=field)
        #print("START FIELD ALIASER: ", field_txt)
        # print(self[field]['aliaser'])
        try:
            alias = self[field]['aliaser'][field_txt]
            parsed = self[field]['parsed'][alias]
            return alias, parsed
        except (AttributeError, KeyError):
            pass
        # print(f"FIELD FROM STR:{txt}")
        parsed = {}
        alias = ''
        if field_txt:
            self.reverse(alias=field_txt, value=txt, field=field)
            ed = self.settings.get('extra_delimiter', ' ')
            base_txt, *extra = field_txt.split(ed, 1)
            subfield_delimiter = self.get_options(key='subfield_delimiter', default=self.settings.sd, field=field)
            prefixes = self.get_options(key='remove', default=[], field=field)
            remove_prefixes = self.get_options(key='remove_prefixes', default=[], field=field)
            named_value_to_str = self[field]['parser']
            prefix_to_str = self[field].get('prefix', {})
            remove_value_if_true = self.get_options(key='remove_value_if_true', field=field)
            remove_prefix_if_str = self.get_options(key='remove_prefix_if_str', field=field)
            remove_prefix_if_false = self.get_options(key='remove_prefix_if_false', field=field)
            alias, parsed = self.remove_subfields_from_str(base_txt, field, prefixes,
                                                           delimiter=subfield_delimiter,
                                                           named_value_to_str=named_value_to_str,
                                                           prefix_to_str=prefix_to_str,
                                                           remove_prefixes=remove_prefixes,
                                                           remove_value_if_true=remove_value_if_true,
                                                           remove_prefix_if_str=remove_prefix_if_str,
                                                           remove_prefix_if_false=remove_prefix_if_false)
            self[field]['aliaser'][base_txt] = alias
            if parsed:
                self[field]['parsed'][alias] = parsed
            self.reverse(alias, base_txt, field=field)
            if extra:
                extra = extra[0]
                try:
                    # optional additional operations
                    extra_alias, extra_parsed = getattr(self, f'{field}{self.ad}aliaser{self.ad}extra')(extra)
                    parsed = parsed.copy()
                    parsed.update(extra_parsed)
                    alias = alias + self.ed + extra_alias
                    extra_parsed['alias'] = extra_alias
                    parsed['extra'] = extra_parsed
                except (AttributeError, KeyError):
                    alias = alias + self.ed + extra
                self[field]['aliaser'][field_txt] = alias
                if parsed:
                    self[field]['parsed'][alias] = parsed
                self.reverse(alias, field_txt, field=field)
        #print("Field, txt, parsed: ", field, txt, parsed)
        return alias, parsed

    def field_from_str(self, txt, field):
        txt = self.get_name(txt)
        if self[field]:
            if f'{field}{self.fd} ' in txt:
                return self.value_from_str(txt, f'{field}{self.fd} ', delimiter='  ').lstrip(f'{self.special_subfield}:')
            starts_with = self.list_remove(list(self[field].keys()), 'caster')[0]
            # print(f"{field} STARTSWITH: ", starts_with)
            return txt if starts_with and txt.startswith(starts_with) else ''
        return ''

    def remove_subfields_from_str(self, txt, field=None, prefixes=None, types_auto=False, **kwargs):
        field = field if field else self.default_field
        if field == self.default_field:
            p_to_parse, types = None, None
        else:
            p_to_parse = self.to_list_str(self[field]['parser'].keys())
            types = [self[field]['caster'][p] for p in p_to_parse]
        parsed = self.to_parsed(txt, prefixes=p_to_parse, types=types, delimiter=self.settings.sd,
                                types_auto=types_auto)
        out_parsed = parsed.copy() if prefixes else parsed
        [parsed.pop(p) for p in (prefixes or [])]
        txt = self.parsed_to_str(parsed, **kwargs) if parsed else ''
        # print("REMOVE OUT ", txt)
        # print("FINAL: ", out_parsed)
        return txt, out_parsed


class Method(Aliaser):
    DEFAULT_OPTIONS = Aliaser.DEFAULT_OPTIONS.copy()
    METHOD_DELIMITER = Aliaser.REVERSAL_PREFIX
    DEFAULT_OPTIONS.update({'eval_remove_prefixes': ['E'],
                            'skip_fields': ['exp']
                            })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self[self.md] = {}
        # self['eval'] = {}

    @property
    def files(self):
        return self['reverser']

    @property
    def md(self):
        return self.METHOD_DELIMITER

    def is_file_method(self, file):
        return self.isfile(file) and self.load_file(file).get('cls_exp_settings') == self.settings.__class__

    def file_filter(self, files):
        return list(filter(self.is_file_method, files))

    def is_alias_method(self, txt):
        if isinstance(txt, str):
            return txt.endswith(self.md + self.name)
        return False

    def to_alias_method(self, alias):
       return alias + self.md + self.name

    @classmethod
    def has_method(cls, txt):
        return cls.METHOD_DELIMITER in txt

    @classmethod
    def split_alias_method(cls, txt):
        cls.raise_error_if(cls.METHOD_DELIMITER not in txt)
        alias, method = txt.rsplit(cls.METHOD_DELIMITER, 1)
        return alias, method

    def alias(self, txt, *args, **kwargs):
        if self.is_alias_method(txt):  # reverse (to file(s))
            alias, method = self.split_alias_method(txt)
            files = self.files[alias]
            return files
        if txt in self.files:
            return self.files[txt]
        if self.isfile(txt) and self.name not in txt:
            #todo:
            print("TODO")
            asdasd
            return ''
        return super().alias(txt, *args, **kwargs)

    def __repr__(self):
        return self.module_obj_to_str(self.__class__.__name__) + '()'

    def __str__(self):
        return '\n'.join([f'{k} Number:{len(self.to_list_str(v))} ' + self.__str__file(v)
                          for k, v in self.files.items()])

    def __str__file(self, files):
        files = self.to_list(files)
        if all([self.file_name(f).startswith('exp_') for f in files]):
            incomplete = []
            for f in files:
                exp = self.exp_load(f, load_model=False)
                seed = exp.settings.seed
                if not exp.data.is_complete:
                    incomplete.append(seed)
            if not incomplete:
                txt = 'Complete: All'
            else:
                txt = f'Incomplete: {incomplete}'
            return txt
        return ''

    @classmethod
    def exp_load(cls, file, load_model=True, run_to_completion=False, verbose=False):
        exp = Manager.exp_load(file, load_model=load_model, run_to_completion=run_to_completion,
                               verbose=verbose)
        return exp

    @classmethod
    def exp_is_complete(cls, file=None, exp=None, load_model=False, run_to_completion=False, verbose=False,
                        **kwargs):
        if not exp:
            exp = cls.exp_file_load(file, load_model=load_model, run_to_completion=run_to_completion,
                                    verbose=verbose)
        return exp.data.is_complete

    def to_exp_dict(self):
        exp_dict = {alias: self.exp_alias_info(alias) for alias in self.files}
        return exp_dict

    def exp_alias_info(self, alias):
        if self.is_alias_method(alias):
            alias, method = self.split_alias_method(alias)
        else:
            method = self.name
        info = {'alias': alias, 'method':method}
        sep_eval = self.get_options(key='sep', field='eval')
        eval, name = alias.split(sep_eval, 1)
        info.update({'eval': eval, 'name': name, 'files': self.files[alias], 'parsed': self.parsed[alias],
                     'options': self._options})
        return info

    # def find_eval_groups(self):
    #     if self.files:
    #         for alias in self.to_list_str(self.files.keys()):
    #             info = self.alias_info(alias)
    #             if info['eval'] not in self['eval']:
    #                 info.pop('eval')
    #                 self['eval'][info['eval']].append(info)

class Methods(Aliaser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = {}

    @classmethod
    def methods_equal(cls, first, second):
        return first.name == second.name

    @classmethod
    def from_file(cls, file, **kwargs):
        data = cls.load_file(file)
        cls_settings = data['cls_exp_settings']
        txt_settings = cls.module_obj_to_str(cls_settings)
        base, name = txt_settings.rsplit('.', 1)
        txt_method = base.replace('settings', 'results') + '.' + name.replace('Settings', 'Method')
        cls_method = cls.str_to_module_obj(txt_method)
        cls_settings = cls.str_to_module_obj(txt_settings)
        # cls.show_sep()
        # print(file)
        # print(cls_settings)
        # print(txt_settings)
        # print(base, name)
        # print(txt_method)
        # print(cls_method)
        # print(cls_settings)
        # cls.show_sep()
        return cls_method(settings=cls_settings(), **kwargs)

    @property
    def dict(self):
        return self._dict

    @property
    def names(self):
        return self.to_list_str(self.dict.keys())

    @property
    def methods(self):
        return self.dict.values()

    @classmethod
    def has_method(cls, txt):
        return Method.has_method(txt)

    @classmethod
    def split_alias_method(cls, txt):
        return Method.split_alias_method(txt)

    def get_method(self, txt):
        if self.has_method(txt):
            alias, name = self.split_alias_method(txt)
            method = self.get_method_from_name(name)
        elif self.isfile(txt):
           method = self.get_method_from_file(txt)
        else:
            raise ValueError('Method cannot be found.')
        return method

    def get_method_from_file(self, file):
        method = self.from_file(file)
        if method.name in self.dict:
            return self.dict[method.name]
        self.dict[method.name] = method
        self[self.default_field].update(method[self.default_field])
        return method

    def __call__(self, name, **kwargs):
        if name in self.dict:
            return self.dict[name]
        return self.alias(name, **kwargs)

    def get_named_method(self, name):
        return self.dict[name]

    def alias(self, txt, **kwargs):
        #print("IN METHODS ALIAS: ", txt)
        if self.has_method(txt):
            alias, name = self.split_alias_method(txt)
            method = self.dict[name]
            # todo: check this is intended, get files full path
            if self.isfile(txt) and 'agg' == self.parent_dir(txt):
                alias = self.file_name(txt).split(method.options['sep'], 1)[-1]
                if len(self.methods) == 1:
                    alias, name = self.split_alias_method(alias)
                self.reverse(alias, value=txt)
                return alias
            else:
                return method.alias(alias, **kwargs) # note: original str is sent
        if self.isfile(txt):
           method = self.get_method_from_file(txt)
           return  method.to_alias_method(method.alias(txt, **kwargs))
        return self[self.default_field].get(txt, txt)

    def __str__(self):
        txt = ''
        for k, v in self.dict.items():
            txt += f'{v.md}{k}\n' +  v.__str__()
        return txt

    def to_exp_dict(self):
        methods = {name: method.to_exp_dict() for name, method in self.dict.items()}
        return methods


class Performance(IO, AttributeFieldDict):
    DEFAULT_FIELDS = ('perf', 'df')
    DEFAULT_REGISTER_PATH = '~/storage/data/boic/register/agg'

    def __init__(self, eval=None, register_path=None):
        super().__init__()
        self._eval = eval
        self._register_path = register_path
        self.df_acq = None
        self.df_global = None
        self.df_raw = None

    @classmethod
    def unique_level_values(cls, df, level: [int, str] = 0, axis=1):
        return cls.df_unique_level_values(df, level=level, axis=axis)

    def alias(self, eval, **kwargs):
        return self.file_name(eval)

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, value):
        self._eval = self.alias(value.rstrip('txt').rstrip('.'))

    @property
    def register_path(self):
        return self.fix_path(self.DEFAULT_REGISTER_PATH if self._register_path is None else self._register_path)

    @property
    def acq_names(self):
        return self.df_unique_level_values(self.df_acq, level='name')

    @property
    def acq_indicators(self):
        return self.df_unique_level_values(self.df_acq, level='indicator')

    @property
    def acq_stats(self):
        return self.df_unique_level_values(self.df_acq, level='stat')

    @property
    def acq_idx(self):
        return self.df_acq.index

    @property
    def global_names(self):
        return self.df_unique_level_values(self.acq, level='name', axis=0)

    @property
    def global_indicators(self):
        return self.df_unique_level_values(self.df_global, level='indicator')

    @property
    def global_stats(self):
        return self.df_unique_level_values(self.acq, level='stat')

    @property
    def names(self):
        return self.acq_names

    @property
    def df_stats(self):
        return self.acq_stats

    @property
    def stats(self):
        return list(self.perf.keys())

    def compute(self, sort=True, sort_by='median', sort_best=True):
        self['ai'] = self.df_global['best_ai']
        self['nai'] = self.df_global['best_nai']
        if sort:
            self['ai'] = self['ai'].sort_values(by=[sort_by], ascending=not sort_best)
            self['nai'] = self['nai'].sort_values(by=[sort_by], ascending=not sort_best)
        self['auc'] = self.slice(label='best_auc', iloc=-1, sort=sort, sort_by=sort_by, ascending=not sort_best)
        return self.perf

    def slice(self, label=None, df=None, level='indicator', axis=1, iloc=None, loc=None, index_label='name',
              pivot_label='stat', sort=True, sort_by='mean', ascending=False):

        _df = self.df_acq if df is None else df
        if label:
            df = pd.DataFrame(_df.xs(label, level=level, axis=axis))
            df.name = label
        else:
            _df = df
        if iloc is None and loc is None:
            return df
        df = pd.DataFrame(df.iloc[-1 if iloc is None else iloc] if loc is None else df.loc[loc])
        df.reset_index(inplace=True)
        df.set_index(index_label, inplace=True)
        df = df.pivot(columns=pivot_label).droplevel(0, axis=1).reindex(
            columns=self.unique_level_values(_df, level=pivot_label))
        if sort:
            df.sort_values(by=[sort_by], ascending=ascending, inplace=True)
        return df

    def load_register(self):
        return self.load_txt(self.join_path(self.register_path, self.add_ext(self.eval)))

    def from_register(self, eval, path=None, verbose=False, **kwargs):
        if path:
            self._register_path = path
        self.eval = eval
        # print(eval)
        return self.from_files(files=self.load_register(), verbose=verbose, **kwargs)

    def from_files(self, files, sort=True, verbose=False, **kwargs):
        self.df_global = None
        self.df_acq = None
        self.df_raw = None
        for file in files:
            if verbose:
                print("LOADING: ", file)
            name = self.alias(file, **kwargs)
            # print(name)
            if not name:
                continue
            data = self.load_file(file)
            data_g = data['global']
            data_g.index = [name]
            data_g.index.name = 'name'
            data_g.columns.names = ['indicator', 'stat']
            self.df_global = data_g if self.df_global is None else pd.concat((self.df_global, data_g))
            data_acq = data['acq']
            indicators = self.unique_level_values(data_acq, level=0)
            stats = self.unique_level_values(data_acq, level=1)
            data_acq.columns = pd.MultiIndex.from_product([[name], indicators, stats],
                                                          names=['name', 'indicator', 'stat'])
            data_acq.index.name = 'acq'
            self.df_acq = data_acq if self.df_acq is None else pd.concat((self.df_acq, data_acq), axis=1)
            data_raw = data['raw_auc']
            runs = data_raw.columns.to_list()
            data_raw.columns = pd.MultiIndex.from_product([[name], ['best_auc'], runs],
                                                          names=['name', 'indicator', 'run'])
            data_raw.index.name = 'acq'
            self.df_raw = data_raw if self.df_raw is None else pd.concat((self.df_raw, data_raw), axis=1)
        if sort:
            self.df_acq.sort_index(axis=1, level='name', sort_remaining=False, inplace=True)
            self.df_raw.sort_index(axis=1, level='name', sort_remaining=False, inplace=True)
            self.df_global.sort_index(inplace=True)
        self.compute(sort=sort, **kwargs)
        return self


class MethodPerformance(Performance):
    BASE_CLS_METHOD = Method

    def __init__(self, method=None, eval=None, register_path=None):
        super().__init__(eval=eval, register_path=register_path)
        if method is None:
            method = self.base_cls_method()
        self.raise_error_if(not isinstance(method, self.cls_method))
        self._method = method

    @property
    def method(self):
        return self._method

    def alias(self, name, **kwargs):
        return  self.method.alias(name, **kwargs)

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, value):
        self._eval = self.get_eval_name(value)

    def get_eval_name(self, txt):
        if self.method.has_method(txt):
            sep = self.method.get_options(key='delimiter', default=' ', field='exp')
            return txt.split(sep, 1)[0]
        return self.alias(txt.rstrip('txt').rstrip('.'))


class AggregatedFrame(Performance):
    DEFAULT_FIELDS = ('df', 'perf')
    DEFAULT_REGISTER_PATH = '~/storage/data/boic/register/agg'
    BASE_CLS_METHODS = Methods

    def __init__(self, methods=None, eval=None, register_path=None,
                 register_filter=None, register_filterfalse=None,
                 file_filter=None, file_filterfalse=None, aliaser=None, **kwargs):
        super().__init__(eval=eval, register_path=register_path)
        if methods is None:
            methods = self.base_cls_methods()
        self.raise_error_if(not isinstance(methods, self.base_cls_methods))
        self._methods = methods
        self.register_filter = register_filter
        self.register_filterfalse = register_filterfalse
        self.file_filter = file_filter
        self.file_filterfalse = file_filterfalse
        self.aliaser = aliaser

    @classmethod
    def filter_options(self, filter=None, filterfalse=None):
        options = {}
        if filter is not None:
            options['filter'] = filter
        if filterfalse is not None:
            options['filterfalse'] = filterfalse
        return options

    @property
    def aliaser(self):
        return self._aliaser

    @aliaser.setter
    def aliaser(self, value):
        self._aliaser = value

    @property
    def methods(self):
        return self._methods

    def alias(self, name, **kwargs):
        # print(f"IN AGG ALIAS: '{name}'")
        alias = self.methods.alias(name, **kwargs) # run this anyway to register name internally
        if self.aliaser: # override alias
            alias = self.aliaser(name, **kwargs)
        return alias

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, value):
        self._eval = self.get_eval_name(value)

    @property
    def file_filter(self):
        return self._file_filter

    @file_filter.setter
    def file_filter(self, value):
        self._file_filter = value

    @property
    def file_filterfalse(self):
        return self._file_filterfalse

    @file_filterfalse.setter
    def file_filterfalse(self, value):
        self._file_filterfalse = value

    @property
    def file_options(self):
        return self.filter_options(filter=self.file_filter, filterfalse=self.file_filterfalse)

    @property
    def register_filter(self):
        return self._register_filter

    @register_filter.setter
    def register_filter(self, value):
        self._register_filter = value

    @property
    def register_filterfalse(self):
        return self._register_filterfalse

    @register_filterfalse.setter
    def register_filterfalse(self, value):
        self._register_filterfalse = value

    @property
    def register_options(self):
        return self.filter_options(filter=self.register_filter, filterfalse=self.register_filterfalse)

    def get_eval_name(self, txt):
        if self.methods.has_method(txt):
            method = self.methods.get_method_from_name(self.methods.split_alias_method(txt)[-1])
            sep = method.get_options(key='delimiter', default=' ', field='exp')
            return txt.split(sep, 1)[0]
        else:
            sep = self.methods.options['sep']
            if sep in txt:
                return txt.split(sep, 1)[0]
        return self.alias(txt.rstrip('txt').rstrip('.'))

    def load_register(self):
        # load and filter files
        return self.to_list(super().load_register(), **self.file_options)


class Results(IO, ParsingDict):
    DEFAULT_FIELDS = ('exp', 'agg')
    IMMUTABLE_AFTER_LOADING = False
    DEFAULT_REGISTER_PATH = '~/storage/data/boic/register/'
    DEFAULT_AGG_SUBPATH = 'agg'
    PENDING_REGISTER_NAME = 'pending'
    EXCLUDED_STATS_KEYS = ['curr_input', 'best_input', 'best_a_1_2_5']
    BASE_CLS_AGG_FRAME = AggregatedFrame

    def __init__(self, *args, register_path=None, agg_register_path=None, cls_agg_frame=None, colors=None,
                 **kwargs):
        super().__init__()
        self._register_path = register_path
        self._agg_register_path = agg_register_path
        if cls_agg_frame is None:
            cls_agg_frame = self.BASE_CLS_AGG_FRAME
        self.raise_error_if(not issubclass(cls_agg_frame, self.BASE_CLS_AGG_FRAME))
        self._cls_agg_frame = cls_agg_frame
        self.agg['af'] = cls_agg_frame(*args, register_path=self.agg_register_path, **kwargs)
        self.argload()
        self['dict'] = {}
        self.colors = colors or {}

    @classmethod
    def aggregate_stats(cls, exp, name, agg_path, agg_len=None, register_name=None, overwrite=False,
                        verbose=False, **kwargs):
        cls.makedir(agg_path)
        agg_file = cls.join_path(agg_path, name)
        if cls.isfile(agg_file):
            if overwrite:
                cls.safe_remove_file(agg_file)
                if register_name:
                    cls.remove_txt(cls.join_path(agg_path, register_name), agg_file)
            else:
                return
        exp['files'] = cls.to_list_str(exp['files'])
        files = cls.list_unique(exp['files'])
        if agg_len:
            if len(files) < agg_len:
                if verbose:
                    print(f"Skipped agg: {name}. Only {len(files)}/{agg_len} files.")
                return
            files = files[:agg_len]
        anp = np.array([f.split('eval_')[-1] for f in exp['files']])
        assert(all(anp == anp[0]))
        res_acq = {}
        res_g = {}
        targets = []
        auc = []
        for i, file in enumerate(files):
            data = cls.load_file(file)['exp_data']['data']
            targets.append(data['train']['targets'])
            auc.append(data['stats']['best_auc'])
            for k in ['time', 'stats']:
                for sk, v in data[k].items():
                    if sk in cls.EXCLUDED_STATS_KEYS or isinstance(v, np.ndarray):
                        continue
                    res = res_acq if isinstance(v, list) else res_g
                    name = f'time_{sk}' if k == 'time' else sk
                    if name in res:
                        res[name].append(v)
                    else:
                        res[name] = [v]
        res = {'global': cls.to_stats_df(res_g), 'acq': cls.to_stats_df(res_acq),
               'targets': pd.DataFrame(np.array(targets).T), 'exp': exp,
               'raw_auc': pd.DataFrame(np.array(auc).T)}
        cls.save_file(agg_file, res)
        if verbose:
            print(f'Saved agg file: {cls.file_name(agg_file)}\n')
        if register_name:
            agg_register = cls.join_path(agg_path, register_name)
            cls.update_txt(agg_register, agg_file)
        return res

    @classmethod
    def filter_options(cls, *args, **kwargs):
        return cls.BASE_CLS_AGG_FRAME.filter_options(*args, **kwargs)

    @classmethod
    def get_registers_from_path(cls, path=None, exclude_pending=True):
        register_path = cls.DEFAULT_REGISTER_PATH if path is None else path
        registers = cls.list_files(register_path, name='*.txt')
        if exclude_pending:
            registers = cls.remove_pending_register(registers)
        return registers

    @classmethod
    def get_register_name(cls, register):
        return cls.file_name(register, ext='txt')

    @classmethod
    def remove_named_register(cls, registers, name):
        return [r for r in registers if not r.rstrip('.txt').endswith(name.rstrip('.txt'))]

    @classmethod
    def remove_pending_register(cls, registers):
        return cls.remove_named_register(registers, name=cls.PENDING_REGISTER_NAME)

    @classmethod
    def plot_df(cls, *args, **kwargs):
        return plot_df(*args, **kwargs)

    @property
    def af(self):
        return self.agg_af

    @property
    def alias(self):
        return self.af.alias

    @property
    def aliaser(self):
        return self.af.aliaser

    @aliaser.setter
    def aliaser(self, value):
        self.af.aliaser = value

    @property
    def agg_register_path(self):
        return self.fix_path(self.join_path(self.register_path, self.DEFAULT_AGG_SUBPATH)
                             if self._agg_register_path is None else self._agg_register_path)

    @property
    def default_commands(self):
        return dict(overwrite=dict(type=self.str_to_bool, default=False),
                    agg_len=dict(type=int, default=10),
                    refactor=dict(type=self.str_to_bool, default=True),
                    verbose=dict(type=self.str_to_bool, default=True))

    @property
    def data(self):
        return self.af

    @property
    def dict(self):
        return self.exp_dict

    @property
    def exp_dict(self):
        if not self['dict']:
            self['dict'] = self.to_exp_dict()
        return self['dict']

    @property
    def evals(self):
        return self.register_names

    @property
    def file_options(self):
        return self.af.file_options

    @property
    def get_commands_extra(self):
        return None

    @property
    def methods(self):
        return self.af.methods

    @property
    def registers(self):
        return self.get_registers_from_path(self.register_path)

    @property
    def register_names(self):
        return list(map(self.get_register_name, self.get_registers_from_path(self.register_path)))

    @property
    def register_path(self):
        return self.fix_path(self.DEFAULT_REGISTER_PATH if self._register_path is None else self._register_path)

    @property
    def register_options(self):
        return self.af.register_options

    @property
    def stats(self):
        return self.af.acq_stats

    def aggregate(self, path=None, overwrite=None, agg_len=10, verbose=None):
        if path:
            self._agg_register_path = path
        if overwrite is not None:
            self['overwrite'] = overwrite
        if agg_len is not None:
            self.agg['len'] = agg_len
        if verbose is not None:
            self['verbose'] = verbose
        agg_path = self.agg_register_path
        overwrite = self['overwrite']
        agg_len = self.agg['len']
        verbose = self['verbose']
        self.fix_registers()
        if self.refactor:
            self.move_exp_files()
        exp_dict = self.exp_dict
        #if verbose:
        #    self.show_exp()
        for method_name, method_exp in exp_dict.items():
            method = self.methods(method_name)
            if verbose:
                print(method.md, method_name)
            for name, exp in method_exp.items():
                if verbose:
                    print("NAME: ", name)
                self.aggregate_stats(exp=exp, agg_path=agg_path, agg_len=agg_len, name=method.to_alias_method(name),
                                     register_name=exp['eval'], overwrite=overwrite, verbose=verbose)

    def fix_registers(self, registers=None):
        registers = self.to_list(self.registers, **self.register_options) if registers is None \
                    else self.to_list(registers)
        for register in registers:
            register_name = self.get_register_name(register)
            exp_files = list(filter(None, self.load_file(register)))
            dirs = self.list_unique(self.to_list(exp_files, map=lambda x: self.split_path(x)[0]))
            new_exp_files = self.to_list(self.collapse([self.list_files(_dir) for _dir in dirs]),
                                         filter=lambda x: register_name in x and self.isfile(x))
            if len(exp_files) != len(new_exp_files):
                print("Fixed: ", register)
                self.save_file(register, sorted(new_exp_files))

    def get_exp_files(self, paths=None, registers=None, register_path=None):
        if paths:
            return {p: self.to_list(self.list_files(p), **self.file_options)
                    for p in self.to_list_str(paths)}
        if registers is None:
            registers = self.to_list(self.get_registers_from_path(register_path), **self.register_options)
        exp_files = {self.get_register_name(r): self.to_list(self.load_txt(r), **self.file_options)
                     for r in self.to_list_str(registers)}
        exp_files = {k: v for k, v in exp_files.items() if len(v) != 0 }
        return exp_files

    def list_agg_registers(self, agg_path=None, names=False):
        if agg_path:
            self._agg_register_path = agg_path
        registers = self.list_files(self.agg_register_path, name='*.txt')
        if names:
            registers = [self.file_name(r, ext='txt') for r in registers]
        return registers

    def line_plot(self, indicator=None, names=None, disp='mean', error_bars=True, colors=None, title=None, **kwargs):
        indicator = indicator or 'best_target_unit_gap'
        colors = colors or self.colors
        title = title or self.eval_name
        pidx = pd.IndexSlice
        df = self.data.slice(indicator)
        df.name = indicator
        if names:
            new_ix = pd.MultiIndex.from_product([self.to_list_str(names),
                                                 self.to_list_str(self.stats)], names=['name', 'stat'])
            name = df.name
            df = df.loc[:, new_ix]
            df.name = name
        self.plot_df(df, title=title, colors=colors, disp=disp, error_bars=error_bars, **kwargs)

    def load(self, eval_name=None, verbose=False):
        if eval_name:
            self.eval_name = eval_name
        self.af.from_register(self.eval_name, verbose=verbose)  # computes perf
        self.raw_auc = self.data.slice('best_auc', df=self.data.df_raw)
        return self

    def move_exp_files(self, registers=None, register_path=None):
        if registers is None:
            registers = self.get_registers_from_path(register_path)
        registers = self.to_list(self.to_list_str(registers), **self.register_options)
        for r in registers:
            register_name, files, changed = self.file_name(r, ext='txt'), [], False
            current_files = self.load_txt(r)
            filtered_file_names = [self.split_path(pfname)[1]
                                   for pfname in self.to_list(current_files, **self.file_options)]
            for f in current_files: # iterate through all
                path, name = self.split_path(f)
                if not path.endswith(register_name) and name in filtered_file_names:
                    changed, file = True, self.join_path(path, register_name, name)
                    f = self.move_file(f, file)
                files.append(f) # to save them later if needed
            if changed:
                self.save_txt(r, files)

    def to_exp_dict(self, paths=None, registers=None, register_path=None):
        exp_files = self.get_exp_files(paths, registers, register_path)
        aliases = [[self.alias(vv) for vv in v] for f, v in self.to_flat_dict(exp_files).items()]
        exp_dict = self.methods.to_exp_dict()
        return exp_dict

    def show_exp(self):
        print(self.methods)