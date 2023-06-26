from boic.core.results import MethodPerformance
from boic.torch.gp.results import GPTorchMethod
from boic.torch.gp.botorch.settings import BoTorchSettings


class BoTorchMethod(GPTorchMethod):
    BASE_CLS_SETTINGS = BoTorchSettings


class BoTorchPerformance(MethodPerformance):
    BASE_CLS_METHOD = BoTorchMethod