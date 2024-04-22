import torch
from torch import Tensor
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel 
from botorch.acquisition import (
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import(
    qLogNoisyExpectedHypervolumeImprovement
)
from botorch.acquisition.utils import prune_inferior_points
from hypermapper.param.transformations import preprocess_parameters_array


class BotorchAcquisitionFunction: # well, basically only the qEI-family

    def __call__( # all the hypermapper kwargs that need to be swallowed
        self,
        settings,
        param_space,
        X,
        objective_weights,
        regression_models,
        iteration_number,
        classification_model,
        feasibility_threshold, 
        **kwargs
    ):
        X, names = preprocess_parameters_array(X, param_space)
        # TODO may want to unsqueeze dim (-2)
        if classification_model is not None:
            feasibility_indicator = classification_model.feas_probability(X)
        else:
            feasibility_indicator = torch.ones(len(X))
        
        acqval = self.acq(X.unsqueeze(-2))
        acqval = (
            acqval * 
            feasibility_indicator * 
            (feasibility_indicator >= feasibility_threshold)
        )
        return acqval


class qLogNEI(BotorchAcquisitionFunction):
    
    acq_class = qLogNoisyExpectedImprovement

    def __init__(self, model):
        model = model
        X_baseline = prune_inferior_points(model, model.train_inputs[0])
        self.acq = self.acq_class(model, X_baseline)


class qLogNEHVI(BotorchAcquisitionFunction):
    
    acq_class = qLogNoisyExpectedHypervolumeImprovement

    def __init__(self, model):
        model = model
        ref_point = torch.zeros(model.train_targets.shape[0])
        self.acq = self.acq_class(model, ref_point, model.train_inputs[0][0], prune_baseline=True)


