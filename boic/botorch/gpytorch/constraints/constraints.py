from gpytorch.constraints.constraints import GreaterThan, softplus, inv_softplus


class Positive(GreaterThan):
    # Added lower_lower bound as argument
    def __init__(self, lower_bound=0.0, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        super().__init__(lower_bound=lower_bound, transform=transform, inv_transform=inv_transform,
                         initial_value=initial_value)

    def __repr__(self):
        return self._get_name() + "()"

    def transform(self, tensor):
        transformed_tensor = self._transform(tensor) if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        tensor = self._inv_transform(transformed_tensor) if self.enforced else transformed_tensor
        return tensor