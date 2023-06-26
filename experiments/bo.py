from boic import Manager, BoTorchSettings

class BoTorchManager(Manager):
    BASE_CLS_EXP_SETTINGS = BoTorchSettings

if __name__ == "__main__":
    m = BoTorchManager()
    m.run()
