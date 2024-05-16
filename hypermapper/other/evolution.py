import itertools
import sys
import time
import numpy as np
from functools import partial
import torch
from hypermapper.param.parameters import (
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    OrdinalParameter,
    PermutationParameter
)
from hypermapper.param import space
from hypermapper.param.data import DataArray
from hypermapper.util.file import (
    initialize_output_data_file,
    load_previous,
)
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.display.multi import MultiObjectiveOutput
from hypermapper.param.doe import get_doe_sample_configurations
from hypermapper.param.constraints import evaluate_constraints
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from pymoo.operators.crossover.ux import UniformCrossover

from hypermapper.util.settings_check import (
    settings_check_evo
)


class FixedInitialSampling(Sampling):

    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples

    def _do(self, problem, n_samples, **kwargs):
        return self.samples
    

class EvolutionAlreadyFinishedError(ValueError):
    pass


def parse_settings_to_evolution(settings, search_space):
    bench_input_parameters = settings["input_parameters"]
    objectives = settings["optimization_objectives"]
    constraints = search_space.constraints
    variables = {}
    variable_map = {}
    inverse_variable_map = {}
    for param_idx, param in enumerate(search_space.parameters):
        param_name = param.name
        if isinstance(param, RealParameter):
            bounds = tuple(param.min_value, param.max_value)
            variables[param_name] = Real(bounds=bounds)
            variable_map[param_name] = lambda x: x
            inverse_variable_map[param_name] = lambda x: x
        
        elif isinstance(param, IntegerParameter):
            bounds = tuple(param.min_value, param.max_value)
            variables[param_name] = Integer(bounds=bounds)
            variable_map[param_name] = lambda x: x
            inverse_variable_map[param_name] = lambda x: x
        
        elif isinstance(param, OrdinalParameter):
            choices = param.values
            param_range = list(range(len(choices)))
            variables[param_name] = Integer(bounds=(param_range[0], param_range[-1]))
            def map_ordinal(x, param):
                return param.values[x]
            variable_map[param_name] = partial(map_ordinal, param=param)
            def invert_ordinal(x, param):
                return param.get_index(x)
            inverse_variable_map[param_name] = partial(invert_ordinal, param=param)

        # sice perms cannot be combined with other variable types,
        # we're forced to have them as a choice variable
        elif isinstance(param, CategoricalParameter):
            variables[param_name] = Choice(options=param.values)
            def map_cat(x, param):
                return param.values[x]
            variable_map[param_name] = partial(map_cat, param=param)
            def invert_cat(x, param):
                return param.get_index(x)
            inverse_variable_map[param_name] = partial(invert_cat, param=param)

        # sice perms cannot be combined with other variable types,
        # we're forced to have them as a choice variable
        elif isinstance(param, PermutationParameter):
            all_perms = list(itertools.permutations(range(param.n_elements)))
            choices = list(range(len(all_perms)))   
            default = search_space.get_default_configurations()
            all_perm_configs = default.repeat(len(choices), 1)
            all_perm_configs[:, param_idx] = torch.Tensor(choices)
            configs = search_space.convert(all_perm_configs, "internal", "original")
            list_of_dict_configs = {p.name: [] for p in search_space.parameters}
            for conf in configs:
                for p_idx, p in enumerate(search_space.parameters):
                    list_of_dict_configs[p.name].append(conf[p_idx])
            
                viable_configs = evaluate_constraints(
                search_space.constraints, 
                list_of_dict_configs,
                return_each_constaint=False
            )
            choices = np.array(choices)[viable_configs]
            
            # TODO filter_conditional_values()
            variables[param_name] = Choice(options=choices)
            def map_perm(x, param):
                return param.get_permutation_value(x)
            variable_map[param_name] = partial(map_perm, param=param)
            def invert_perm(x, param):
                return param.get_int_value(x)
            inverse_variable_map[param_name] = partial(invert_perm, param=param)
        else:
            raise ValueError("No such param type.")

    return {
        "variables": variables, 
        "variable_map": variable_map, 
        "inverse_variable_map": inverse_variable_map, 
        "constraints": constraints, 
        "objectives": objectives,
        "param_names": list(bench_input_parameters.keys())
    }


class BlackBoxEvolutionProblem(ElementwiseProblem):

    def __init__(self, evo_kwargs, function: callable, max_queries: int):
        vars = evo_kwargs["variables"]
        super().__init__(
            vars=evo_kwargs["variables"],
            n_obj=len(evo_kwargs["objectives"]),
        )
        
        self.objectives = evo_kwargs["objectives"]
        self.black_box_function = function
        self.variable_map = evo_kwargs["variable_map"]
        self.inverse_variable_map = evo_kwargs["inverse_variable_map"]
        
        #self.has_constraints = True
        self.constraints = evo_kwargs["constraints"]
        self.param_names = evo_kwargs["param_names"]
        self.n_ieq_constr = 1
        self.n_eq_constr = len(self.constraints)
        self.max_queries = max_queries
        self.current_queries = 0


    def _evaluate(self, x, out, *args, **kwargs):
        # only takes one config at a time
        print(x)
        if self.current_queries == self.max_queries:
            raise EvolutionAlreadyFinishedError
        x_reparam = [self.variable_map[name](x_val) for name, x_val in x.items()]
        dict_config = {name: [val] for name, val in zip(self.param_names, x_reparam)}
        
        viable_configs = evaluate_constraints(
            self.constraints, 
            dict_config,
            return_each_constaint=True
        )
        viable = np.all(viable_configs)
        if viable:
            # if we know a priori that the config is viable, we query the black box
            res = self.black_box_function([x_reparam])
            out["F"] = res.metrics_array.flatten().tolist()
            out["G"] = (2 * res.feasible_array.flatten().to(torch.long) -1).tolist()
            out["H"] = (viable_configs).flatten().tolist()
            self.current_queries += 1
        else:
            out["H"] = (viable_configs).flatten().tolist()
            out["F"] = [9999] * len(self.objectives) 
            out["G"] = [-1]
        

def main(settings, black_box_function=None):
    """
    Run design-space exploration using bayesian optimization.
    Input:
        - settings: dictionary containing all the configuration parameters of this optimization.
        - black_box_function: the black box function to optimize (not needed in client-server mode).
    Returns:
        a DataArray object containing the data collected during the exhasutive search.
    """
    start_time = time.time()
    settings = settings_check_evo(settings, black_box_function)
    param_space = space.Space(settings)
    initialize_output_data_file(settings, param_space.all_names)
    
    data_array = DataArray(
        torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
    )
    absolute_configuration_index = 0

    beginning_of_time = param_space.current_milli_time()
    if settings["resume_optimization"]:
        data_array, absolute_configuration_index, beginning_of_time = load_previous(
            param_space, settings
        )
        space.write_data_array(param_space, data_array, settings["output_data_file"])
    doe_parameter_array = torch.Tensor()
    
    default_parameter_array = torch.Tensor()
    default_configurations = param_space.get_default_configurations()
    if (
        absolute_configuration_index
        < settings["design_of_experiment"]["number_of_samples"]
    ):
        doe_parameter_array = get_doe_sample_configurations(
            param_space,
            data_array,
            settings["design_of_experiment"]["number_of_samples"]
            - absolute_configuration_index,
            settings["design_of_experiment"]["doe_type"],
            allow_repetitions=settings["design_of_experiment"]["allow_repetitions"],
        )

    default_doe_parameter_array = torch.cat(
        (
            default_parameter_array.reshape(-1, param_space.dimension),
            doe_parameter_array.reshape(-1, param_space.dimension),
        ),
        0,
    )
    def evo_black_box(config): 
        config = param_space.convert(config, "original", "internal")
        new_data_array = param_space.run_configurations(
            config,
            beginning_of_time,
            settings,
            black_box_function,
        )
        return new_data_array
    
    max_queries = settings["optimization_iterations"] + settings["design_of_experiment"]["number_of_samples"]
    evo_kwargs = parse_settings_to_evolution(settings=settings, search_space=param_space)
    evo_problem = BlackBoxEvolutionProblem(evo_kwargs=evo_kwargs, function=evo_black_box, max_queries=max_queries)
    
    survival = RankAndCrowdingSurvival()
    off = UniformCrossover(prob=1.0)
    initial_array = param_space.convert(default_doe_parameter_array, "internal", "original")
    initial_samples = []
    param_names = evo_kwargs["param_names"]
    inv_varmap = evo_kwargs["inverse_variable_map"]
    
    for config in initial_array:
        conf = {}
        for p_name, val in zip(param_names, config):
            conf[p_name] = inv_varmap[p_name](val)
        initial_samples.append(conf)

    sampling = FixedInitialSampling(initial_samples)
    algorithm = MixedVariableGA(
        sampling=sampling, 
        output=MultiObjectiveOutput(), 
        survival=survival)
    try:
        res = minimize(
            evo_problem,
            algorithm,
            get_termination("n_eval", 2500000),
            sampling=sampling,
            seed=settings["seed"],
            verbose=False
        )
    except EvolutionAlreadyFinishedError:
        pass

    print("End of evolutionary search\n")

    sys.stdout.write_to_logfile(
        ("Total script time %10.2f sec\n" % (time.time() - start_time))
    )

    return data_array
