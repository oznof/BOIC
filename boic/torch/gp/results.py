from boic.torch.results import TorchMethod
from .settings import GPTorchSettings


class GPTorchMethod(TorchMethod):
    BASE_CLS_SETTINGS = GPTorchSettings
    # DEFAULT_TRAIN = TorchAliaser.DEFAULT_TRAIN.copy()
    # DEFAULT_TRAIN.update({'M': {'lbfgsb': ''}, 'MM': {'botorch': ''}})
    # DEFAULT_ACQ = TorchAliaser.DEFAULT_ACQ.copy()
    # DEFAULT_ACQ.update({'F': {'ei': ''}, 'M': {'lbfgsb': ''}, 'MM': {'botorch': ''}})
    # DEFAULT_MODEL = {'MM': {'const': ''}, 'BK': {'matern': ''}, 'BKM': {'ard': ''}, 'KM': {'fixed': ''} }
    # DEFAULT_OPTIONS = TorchAliaser.DEFAULT_OPTIONS.copy()
    # DEFAULT_OPTIONS.update({'model_starts_with': 'MM'})
    DEFAULT_MODEL = {'parser': {'K': {'sk': 'sk'}, 'KM': {'base': ''}}}

    def model_aliaser_extra(self, txt):
        parsed = {}
        if txt.startswith('P'):
            alias, parsed = self.remove_subfields_from_str(txt,
                                                           named_value_to_str={'P': {False: '', True: ''}},
                                                           order='reverse', types_auto=True)
            return alias, parsed
        return txt, parsed
