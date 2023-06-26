from pulse.jobs import IO
from pulse.jobs import RegisterIO as PRegisterIO
from pulse.jobs import PendingIO as PPendingIO
from pulse.jobs import Job as PJob


class RegisterIO(PRegisterIO):
    PKG_NAME = 'boic'
    DEFAULT_PATH = f'~/storage/data/{PKG_NAME}/'

class PendingIO(PPendingIO):
    PKG_NAME = 'boic'
    DEFAULT_PATH = f'~/storage/data/{PKG_NAME}/'

class Job(PJob):
    PKG_NAME = 'boic'
    DEFAULT_PATH = f'~/storage/data/{PKG_NAME}/'
