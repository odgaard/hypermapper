import sys
from typing import Dict, Any, Union

from gpytorch.models import ExactGP
import botorch.models
import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior, LogNormalPrior
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from hypermapper.bo.models.kernels import (
    TransformedOverlapKernel,
    MallowsKernel,
    WeightedAdditiveKernel,
)
from hypermapper.param.space import get_permutation_settings
import warnings

from hypermapper.bo.models.models import Model


class GpBotorch(botorch.models.SingleTaskGP, Model):
    """
    A wrapper for the botorch GP https://botorch.org/.
    """

    def __init__(
        self,
        settings,
        X: torch.Tensor,
        y: torch.Tensor,
        param_dims: Dict[str, list],
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y training data
        """
        y = y.to(X)
        numerical_dims = param_dims["numerical"]
        permutation_dims = param_dims["permutation"]
        categorical_dims = param_dims["categorical"]
        
        if len(permutation_dims) > 0:
            perm_settings = get_permutation_settings(settings)
        if (
            settings["noise_prior"] is not None
            and settings["noise_prior"]["name"].lower() == "gamma"
        ):
            noise_prior = GammaPrior(*settings["noise_prior"]["parameters"])
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        elif (
            settings["noise_prior"] is not None
            and settings["noise_prior"]["name"].lower() == "lognormal"
        ):
            noise_prior = LogNormalPrior(*settings["noise_prior"]["parameters"])
            noise_prior_mode = None
        else:
            raise ValueError("Noise prior is unavailable, shouldn't be in schema.")

        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=torch.Size(),
            noise_constraint=GreaterThan(
                1e-4,
            ),
        )

        ExactGP.__init__(
            self, train_inputs=X, train_targets=y, likelihood=likelihood)
        if settings["lengthscale_prior"]["name"].lower() == "gamma":
            alpha = float(settings["lengthscale_prior"]["parameters"][0])
            beta = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["lengthscale_prior"]["name"].lower() == "lognormal":
            mu = float(settings["lengthscale_prior"]["parameters"][0])
            sigma = float(settings["lengthscale_prior"]["parameters"][1])
            lengthscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            lengthscale_prior = GammaPrior(3.0, 6.0)

        """
        Outputscale priors and constraints
        """
        # define outputscale priors
        if settings["outputscale_prior"]["name"].lower() == "gamma":
            alpha = float(settings["outputscale_prior"]["parameters"][0])
            beta = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = GammaPrior(concentration=alpha, rate=beta)

        elif settings["outputscale_prior"]["name"].lower() == "lognormal":
            mu = float(settings["outputscale_prior"]["parameters"][0])
            sigma = float(settings["outputscale_prior"]["parameters"][1])
            outputscale_prior = LogNormalPrior(loc=mu, scale=sigma)

        else:
            outputscale_prior = GammaPrior(2.0, 0.15)

        """
        Initialise the kernel
        """
        # mean_module = gpytorch.means.ZeroMean()
        mean_module = gpytorch.means.ConstantMean()

        self.ard_size = X.shape[-1]
        perm_settings = get_permutation_settings(settings)
        if perm_settings in ["spearman", "hamming", "kendall"]:
            num_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                ard_num_dims=len(numerical_dims + categorical_dims),
                active_dims=numerical_dims + categorical_dims,
            )
            perm_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                ard_num_dims=(len(permutation_dims)),
                active_dims=permutation_dims,
            )
            base_kernel = num_kernel * perm_kernel

        elif perm_settings == "mallows":
            num_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                ard_num_dims=len(numerical_dims + categorical_dims),
                active_dims=numerical_dims + categorical_dims,
            )                
            
            perm_kernel = MallowsKernel(
                lengthscale_prior=lengthscale_prior,    
                active_dims=permutation_dims,
            )
        
        elif perm_settings == "transformed_overlap":
            num_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                ard_num_dims=len(numerical_dims + categorical_dims),
                active_dims=numerical_dims + categorical_dims,
            )                
            
            perm_kernel = TransformedOverlapKernel(
                lengthscale_prior=lengthscale_prior,
                active_dims=permutation_dims,
            )
        
        if settings["parameterize_lambda"]:
            base_kernel = WeightedAdditiveKernel(perm_kernel, num_kernel)
        else:
            base_kernel = num_kernel * perm_kernel
            
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
        )

        with warnings.catch_warnings(
            record=True
        ) as w:  # This catches the non-normalized due to default outside of parameter range warning
            warnings.filterwarnings(
                "default", category=botorch.exceptions.InputDataWarning
            )
            botorch.models.SingleTaskGP.__init__(
                self,
                X,
                y.unsqueeze(1),
                likelihood=likelihood,
                covar_module=covar_module,
                mean_module=mean_module,
            )
            for warning in w:
                sys.stdout.write_to_logfile(f"WARNING: {str(warning.message)}\n")

        self.eval()

    def apply_hyperparameters(self, lengthscale, outputscale, noise, mean):
        if not (
            type(lengthscale) is torch.Tensor
            and type(outputscale) is torch.Tensor
            and type(noise) is torch.Tensor
            and type(mean) is torch.Tensor
        ):
            raise TypeError("Hyperparameters must be torch tensors")
        self.covar_module.base_kernel.lengthscale = lengthscale.to(dtype=torch.float64)
        self.covar_module.outputscale = outputscale.to(dtype=torch.float64)
        self.likelihood.noise_covar.noise = noise.to(dtype=torch.float64)
        if isinstance(self.mean_module, gpytorch.means.ConstantMean):
            self.mean_module.constant = mean.to(dtype=torch.float64)

    def fit(
        self,
        settings: Dict[str, Any],
        previous_hyperparameters: Union[Dict[str, Any], None],
    ):
        """
        Fits the model hyperparameters.
        Input:
            - settings:
            - previous_hyperparameters: Hyperparameters of the previous model.
        Returns:
            - Hyperparameters of the model or None if the model is not fitted.
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        fit_gpytorch_mll(mll)
        return None


    def get_mean_and_std(self, normalized_data, predict_noiseless):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: not used for this model.
            - use_var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        if predict_noiseless:
            prediction = self.posterior(normalized_data, observation_noise=False)
        else:
            prediction = self.posterior(normalized_data, observation_noise=True)

        mean = prediction.mean.reshape(-1)
        var = prediction.variance.reshape(-1)
        if any(var < -1e12):
            raise Exception(f"GP prediction resulted in negative variance {var}")
        var += 1e-12

        std = torch.sqrt(var)

        return mean, std
