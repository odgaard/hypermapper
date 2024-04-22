from linear_operator.operators import LinearOperator
import torch
from torch import Tensor
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from gpytorch.priors import NormalPrior


def compute_concordant_pairs(x1: Tensor, x2: Tensor) -> Tensor:
    """Computes the number of concordant and discordant pairs
    in the permutation tensors.

    Args:
        x1 (Tensor): batch_shape x num_perms tensor of permutations.
        x2 (Tensor): batch_shape x num_perms tensor of permutations.

    Returns:
        batch_shape x batch_shape tensor of the number of
        concordant pairs.
    """
    x1 = x1.argsort(-1).unsqueeze(-2)
    x2 = x2.argsort(-1).unsqueeze(-3)
    
    order_diffs = ((x1.unsqueeze(-2) - x1.unsqueeze(-1))
            * (x2.unsqueeze(-2) - x2.unsqueeze(-1))
    )
    #print("Tril start")
    concordant_pairs = (order_diffs.tril() > 0).sum(dim=[-1, -2])
    #print("Tril done, RIP")
    return concordant_pairs


class MallowsKernel(Kernel):
    """Implementation of the Mallows kernel.
    This kernel measures the number of discordant pairs, and 
    computes a correlation from it.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(MallowsKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:

        max_pairs = (x1.shape[-1] * (x1.shape[-1] - 1)) / 2
        #print(x1.shape, x2.shape)
        # TODO diag here already!   
        concordant_pairs = compute_concordant_pairs(x1, x2)
        discordant_pairs = max_pairs - concordant_pairs

        if diag:
            return torch.diagonal(torch.exp(-discordant_pairs / self.lengthscale), dim1=-1, dim2=-2)
        
        return torch.exp(-discordant_pairs / (max_pairs * self.lengthscale))
    


class TransformedOverlapKernel(Kernel):
    """Implementation of the Transformed Overlap kernel.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(TransformedOverlapKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        max_pairs = (x1.shape[-1] * (x1.shape[-1] - 1)) / 2
        concordant_pairs = compute_concordant_pairs(x1, x2)
        if diag:
            return torch.diagonal(
                torch.exp((concordant_pairs == max_pairs).to(x1)  / self.lengthscale),
            dim1=-1, dim2=-2)
        return torch.exp((concordant_pairs == max_pairs).to(x1)  / self.lengthscale)


class WeightedAdditiveKernel(Kernel):
    
    """Implementation of the CoCaBO weighted additive kernel.
    """

    has_lengthscale = False

    def __init__(self, kernel_1: Kernel, kernel_2: Kernel, **kwargs):
        super(WeightedAdditiveKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.register_parameter(name="aug_lambda", parameter=torch.nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            name="aug_lambda_prior", 
            prior=NormalPrior(0, 2), 
            param_or_closure=self._aug_lambda_param, 
            setting_closure=self._aug_lambda_closure)

    def _aug_lambda_param(self, m):
        return m.aug_lambda
    
    def _aug_lambda_closure(self, m, v):
        return m._set_aug_lambda(v)
        
    def _set_aug_lambda(self, value):
        self.aug_lambda = torch.nn.Parameter(torch.as_tensor(value))


    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor | LinearOperator:
        K1 = self.kernel_1(x1=x1, x2=x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        K2 = self.kernel_2(x1=x1, x2=x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        lamb = torch.sigmoid(self.aug_lambda)
        return (1 - lamb) * K1 * K2 + lamb * (K1 + K2) 