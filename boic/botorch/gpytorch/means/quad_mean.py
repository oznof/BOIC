import torch

from gpytorch.means import Mean
from ..utils import make_property

class QuadMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(),
                 a_prior=None, a_constraint=None,
                 constant_prior=None, constant_constraint=None,
                 anchor=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.batch_shape = batch_shape
        self.anchor = anchor
        self.register_parameter(name='raw_a', parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size)))
        if a_prior:
            self.register_prior('a_prior', a_prior, 'a')
        if a_constraint:
            self.register_constraint('raw_a', a_constraint)
        self.register_parameter(name='raw_constant', parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if constant_prior:
            self.register_prior('constant_prior', constant_prior, 'constant')
        if constant_constraint:
            self.register_constraint('raw_constant', constant_constraint)
        self.a = 1

    a = make_property('raw_a')
    constant = make_property('raw_constant')

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        if value is None:
            value = 0
        if isinstance(value, int) or isinstance(value, float):
            value = float(value) * torch.ones(*self.batch_shape, 1, self.input_size)
        value = torch.as_tensor(value)
        if len(value) == 1:
            value *= torch.ones(self.input_size)
        self._anchor = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.anchor.expand_as(x).to(x)
        m_a = x.pow(2).matmul(self.a.unsqueeze(-1))  # b x n x 1
        # m_b = x.matmul(self.b.unsqueeze(-1))
        # (m_a + m_b).squeeze() + self.constant  # b x n, broadcast c along n
        return m_a.squeeze() + self.constant