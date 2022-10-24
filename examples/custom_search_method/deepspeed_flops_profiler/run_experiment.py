"""
This script runs a custom SearchMethod and LocalSearchRunner on
your local machine, while the multi-trial experiment for hyperparameter
search is executed on a Determined cluster.

LocalSearchRunner is responsible for:
 -> executing the custom SearchMethod on a local machine,
 -> creating a multi-trial experiment on the Determined cluster,
 -> handling communication between the multi-trial experiment and your local SearchMethod,
 -> enabling fault tolerance for SearchMethods that implement save_method_state() and
    load_method_state().

LocalSearchRunner receives SearcherEvents from the remote experiment, and passes
the events to your custom SearchMethod, which, in turn, produces a list of Operations in
response to the events. LocalSearchRunner sends the operations to the remote experiment
for execution.
"""
import sys

sys.path.append(".")

import logging
from asha import ASHASearchMethod
from pathlib import Path
from determined import searcher

from deepspeed.profiling.flops_profiler import get_model_profile

from model_def import create_model
import utils

MIN_FLOPS = 1_250_000_000

def filter(hp: utils.HPPoint) -> bool:
    model = create_model(hp)
    flops, _, _ = get_model_profile(
        model=model,
        input_shape=(hp["global_batch_size"], 1, 28, 28),
        print_profile=False,
        detailed=False,
        as_string=False,
    )
    if flops >= MIN_FLOPS:
    #    print(f"Accepted HP configuration for a model with FLOPS={flops} >= {MIN_FLOPS=}")
        return True
    #print(f"Rejected HP configuration for a model with FLOPS={flops} < {MIN_FLOPS=}")
    return False


def sample_params() -> utils.HPPoint:
    return utils.filtered_params(filter)


def main():
    ########################################################################
    # Multi-trial experiment
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # The content of the following directory is uploaded to Determined cluster.
    # It should include all files necessary to run the experiment (as usual).
    model_context_dir = "."

    # Path to the .yaml file with the multi-trial experiment configuration.
    model_config = f"{model_context_dir}/config.yaml"

    ########################################################################
    # Fault Tolerance for LocalSearchRunner
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # To support fault tolerance, LocalSearchRunner saves information required to
    # resume LocalSearchRunner and SearchMethod in the case of unexpected process termination.
    # The required information is stored in LocalSearchRunner.searcher_dir and includes
    # experiment id, LocalSearchRunner state, and SearchMethod state.
    #
    # While LocalSearchRunner saves its own state and ensures invoking save() and
    # load() methods when necessary, a user is responsible for implementing
    # SearchMethod.save_method_state() and SearchMethod.load_method_state() to ensure correct
    # resumption of the SearchMethod.
    searcher_dir = Path("searcher_dir")

    # Instantiate your implementation of SearchMethod
    search_method = ASHASearchMethod(
        search_space=sample_params,
        max_length=1000,
        max_trials=16,
        num_rungs=3,
        divisor=4,
    )

    # Instantiate LocalSearchRunner
    search_runner = searcher.LocalSearchRunner(search_method, searcher_dir=searcher_dir)

    ########################################################################
    # Run LocalSearchRunner
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # 1) Creates new experiment or loads state:
    #      -> if searcher_dir contains information required for resumption
    #         (e.g., experiment id), then LocalSearchRunner loads its own state
    #         and invokes SearchMethod.load_method_state() to restore SearchMethod state;
    #      -> otherwise, new experiment is created.
    # 2) Handle communication between the remote experiment and the custom SearchMethod
    # 3) Exits when the experiment is completed.
    experiment_id = search_runner.run(model_config, model_dir=model_context_dir)
    logging.info(f"Experiment {experiment_id} has been completed.")


if __name__ == "__main__":
    #print(sample_params())
    main()
