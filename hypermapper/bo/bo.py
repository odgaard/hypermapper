import random
import sys
import time

import torch

from hypermapper.bo.models import models
from hypermapper.bo.optimize import optimize_acq
from hypermapper.param import space
from hypermapper.param.data import DataArray
from hypermapper.param.doe import get_doe_sample_configurations
from hypermapper.param.sampling import random_sample
from hypermapper.param.transformations import preprocess_parameters_array
from hypermapper.util.file import load_previous
from hypermapper.util.file import (
    initialize_output_data_file,
)
from hypermapper.util.settings_check import settings_check_bo
from hypermapper.util.util import (
    sample_weight_flat,
    update_mean_std,
)


def main(settings, black_box_function=None):
    """
    Run design-space exploration using bayesian optimization.

    Input:
        - settings: dictionary containing all the settings of this optimization.
        - black_box_function: a name for the file used to save the dse results.
    """

    ################################################
    # SETUP
    ################################################
    start_time = time.time()
    settings = settings_check_bo(settings, black_box_function)
    param_space = space.Space(settings)
    initialize_output_data_file(settings, param_space.all_names)

    if "feasible_output" in settings:
        enable_feasible_predictor = settings["feasible_output"][
            "enable_feasible_predictor"
        ]
    else:
        enable_feasible_predictor = False

    data_array = DataArray(
        torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
    )
    absolute_configuration_index = 0
    ################################################
    # RESUME PREVIOUS
    ################################################
    beginning_of_time = param_space.current_milli_time()
    doe_t0 = time.time()
    if settings["resume_optimization"]:
        data_array, absolute_configuration_index, beginning_of_time = load_previous(
            param_space, settings
        )
        space.write_data_array(param_space, data_array, settings["output_data_file"])
        absolute_configuration_index = data_array.len
    ################################################
    # DEFAULT
    ################################################
    default_parameter_array = torch.Tensor()
    default_configurations = param_space.get_default_configurations()
    if default_configurations is not None:
        for default_configuration in default_configurations:
            if (
                absolute_configuration_index
                >= settings["design_of_experiment"]["number_of_samples"]
            ):
                break
            if (
                not param_space.get_unique_hash_string_from_values(
                    default_configuration
                )
                in data_array.string_dict
            ):
                default_parameter_array = torch.cat(
                    (
                        default_parameter_array.reshape(-1, param_space.dimension),
                        default_configuration.reshape(-1, param_space.dimension),
                    ),
                    0,
                )
                absolute_configuration_index = default_parameter_array.shape[0]
    ################################################
    # DOE
    ################################################
    doe_parameter_array = torch.Tensor()
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
    if default_doe_parameter_array.shape[0] > 0:
        if settings["hypermapper_mode"]["mode"] == "stateless":
            default_original = param_space.convert(
                default_doe_parameter_array, "internal", "original"
            )
            sys.stdout.write_to_logfile(
                f"Stateless mode, returning default and doe configurations\n{default_original}"
            )
            return (
                default_original,
                param_space.parameter_names,
            )
        else:
            default_doe_data_array = param_space.run_configurations(
                default_doe_parameter_array,
                beginning_of_time,
                settings,
                black_box_function,
            )
    else:
        default_doe_data_array = DataArray(
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
    data_array.cat(default_doe_data_array)
    absolute_configuration_index = data_array.len
    iteration_number = max(
        absolute_configuration_index
        - settings["design_of_experiment"]["number_of_samples"]
        + 1,
        1,
    )

    # If we have feasibility constraints, we must ensure we have at least one feasible sample before starting optimization
    # If this is not true, continue design of experiment until the condition is met
    if enable_feasible_predictor:
        while (
            torch.sum(data_array.feasible_array) <= 1
            and settings["optimization_iterations"] >= iteration_number
        ):
            print(
                "Warning: all points are invalid, random sampling more configurations."
            )
            print("Number of samples so far:", absolute_configuration_index)
            tmp_parameter_array = get_doe_sample_configurations(
                param_space,
                data_array,
                1,
                "random sampling",
                allow_repetitions=settings["design_of_experiment"]["allow_repetitions"],
            )
            if settings["hypermapper_mode"]["mode"] == "stateless":
                return (
                    param_space.convert(tmp_parameter_array, "internal", "original"),
                    param_space.parameter_names,
                )
            else:
                tmp_data_array = param_space.run_configurations(
                    tmp_parameter_array, beginning_of_time, settings, black_box_function
                )
            data_array.cat(tmp_data_array)
            absolute_configuration_index += 1
            iteration_number += 1

        if True not in data_array.feasible_array:
            raise Exception("Budget spent without finding a single feasible solution.")

    feasible_data_array = data_array.get_feasible()
    objective_means, objective_stds = update_mean_std(
        feasible_data_array.metrics_array, settings
    )
    print(
        "\nEnd of doe/resume phase, the number of evaluated configurations is: %d\n"
        % absolute_configuration_index
    )
    sys.stdout.write_to_logfile(
        "End of DoE - Time %10.4f sec\n" % (time.time() - doe_t0)
    )

    if data_array.len == 0:
        raise Exception("Cannot run Hypermapper without any initial data.")
    ################################################
    # MAIN LOOP
    ################################################
    bo_t0 = time.time()
    if settings["time_budget"] > 0:
        print(
            "starting optimization phase, limited to run for ",
            settings["time_budget"],
            " minutes",
        )
    elif settings["time_budget"] == 0:
        print("Time budget cannot be zero. To not limit runtime set time_budget = -1")
        sys.exit()
    model_hyperparameters = None

    # loop
    for iteration in range(iteration_number, settings["optimization_iterations"] + 1):
        print(
            "Starting optimization iteration:", iteration, "sample no:", data_array.len
        )
        iteration_t0 = time.time()
        if random.uniform(0, 1) > settings["epsilon_greedy_threshold"]:
            tmp_data_array = (
                data_array.copy()
            )  # copy that will contain fantasized data if batch_size > 1
            tmp_feasible_data_array = tmp_data_array.get_feasible()
            best_configurations = torch.Tensor()
            for batch_idx in range(settings["batch_size"]):
                #############
                # Fit models
                #############
                model_t0 = time.time()
                (
                    regression_models,
                    model_hyperparameters,
                ) = models.generate_mono_output_regression_models(
                    settings=settings,
                    data_array=tmp_feasible_data_array,
                    param_space=param_space,
                    objective_means=objective_means,
                    objective_stds=objective_stds,
                    previous_hyperparameters=model_hyperparameters,
                    reoptimize=(iteration - 1)
                    % settings["reoptimise_hyperparameters_interval"]
                    == 0,
                )
                ##########
                # optimize
                ##########
                if regression_models is None:
                    sys.stdout.write_to_logfile(
                        "Warning: the model failed to fit, random sampling more configurations."
                    )
                    best_configurations = random_sample(
                        param_space=param_space,
                        n_samples=1,
                        allow_repetitions=True,
                        previously_run=data_array.string_dict,
                    )
                else:
                    classification_model = None
                    if enable_feasible_predictor and False in data_array.feasible_array:
                        classification_model = models.generate_classification_model(
                            settings,
                            param_space,
                            data_array,
                        )
                    model_t1 = time.time()
                    sys.stdout.write_to_logfile(
                        "Model fitting time %10.4f sec\n" % (model_t1 - model_t0)
                    )

                    objective_weights = sample_weight_flat(
                        settings["optimization_objectives"]
                    )[0]
                    local_search_t0 = time.time()
                    best_values = torch.min(
                        tmp_feasible_data_array.metrics_array, dim=0
                    )[0]
                    best_configuration = optimize_acq(
                        settings=settings,
                        param_space=param_space,
                        data_array=tmp_data_array,
                        regression_models=regression_models,
                        iteration_number=iteration,
                        objective_weights=objective_weights,
                        objective_means=objective_means,
                        objective_stds=objective_stds,
                        best_values=best_values,
                        classification_model=classification_model,
                    )
                    best_configurations = torch.cat(
                        (best_configurations, best_configuration.unsqueeze(0)), 0
                    )
                    if batch_idx < settings["batch_size"] - 1:
                        preprocessed_best_configuration = preprocess_parameters_array(
                            best_configuration.unsqueeze(0), param_space
                        )
                        fantasized_values = torch.tensor(
                            [
                                model.get_mean_and_std(
                                    preprocessed_best_configuration[0].unsqueeze(0),
                                    False,
                                )[0]
                                for model in regression_models
                            ]
                        )
                        fantasized_data_array = DataArray(
                            best_configuration.unsqueeze(0),
                            fantasized_values.unsqueeze(0),
                            torch.Tensor(0),
                            torch.Tensor(([1] if enable_feasible_predictor else [])),
                        )
                        tmp_data_array.cat(fantasized_data_array)
                        tmp_feasible_data_array = tmp_data_array.get_feasible()
                        objective_means, objective_stds = update_mean_std(
                            feasible_data_array.metrics_array, settings
                        )

                    local_search_t1 = time.time()
                    sys.stdout.write_to_logfile(
                        "Local search time %10.4f sec\n"
                        % (local_search_t1 - local_search_t0)
                    )
        else:
            sys.stdout.write_to_logfile(
                f"random sampling a configuration to run due to epsilon greedy.\n"
            )
            best_configurations = random_sample(
                param_space,
                n_samples=settings["batch_size"],
                allow_repetitions=True,
                previously_run=data_array.string_dict,
            )
        ##################
        # Evaluate configs
        ##################
        black_box_function_t0 = time.time()
        if settings["hypermapper_mode"]["mode"] == "stateless":
            return (
                param_space.convert(best_configurations, "internal", "original"),
                param_space.parameter_names,
            )
        else:
            new_data_array = param_space.run_configurations(
                best_configurations,
                beginning_of_time,
                settings,
                black_box_function,
            )
        data_array.cat(new_data_array)
        feasible_data_array = data_array.get_feasible()
        black_box_function_t1 = time.time()
        sys.stdout.write_to_logfile(
            "Black box function time %10.4f sec\n"
            % (black_box_function_t1 - black_box_function_t0)
        )
        objective_means, objective_stds = update_mean_std(
            feasible_data_array.metrics_array, settings
        )
        run_time = (time.time() - start_time) / 60
        iteration_t1 = time.time()
        sys.stdout.write_to_logfile(
            "Total iteration time %10.4f sec\n" % (iteration_t1 - iteration_t0)
        )
        if run_time > settings["time_budget"] != -1:
            break

    sys.stdout.write_to_logfile(
        "End of BO phase - Time %10.4f sec\n" % (time.time() - bo_t0)
    )
    print("End of Bayesian Optimization")

    sys.stdout.write_to_logfile(
        ("Total script time %10.2f sec\n" % (time.time() - start_time))
    )

    return data_array
