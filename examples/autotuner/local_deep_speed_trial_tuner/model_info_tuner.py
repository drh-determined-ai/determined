import logging
import uuid
from typing import List

from determined import searcher


class SingleSearchMethod(searcher.SearchMethod):
    def __init__(self, experiment_config: dict, max_length: int) -> None:
        # since this is a single trial the hyperparameter space comprises a single point
        self.hyperparameters = experiment_config["hyperparameters"]
        self.max_length = max_length
        self.trial_closed = False

    def on_trial_created(
        self, _: searcher.SearcherState, __: uuid.UUID
    ) -> List[searcher.Operation]:
        return []

    def on_validation_completed(
        self, _: searcher.SearcherState, request_id: uuid.UUID, metric: float, train_length: int
    ) -> List[searcher.Operation]:
        return []

    def on_trial_closed(
        self, _: searcher.SearcherState, request_id: uuid.UUID
    ) -> List[searcher.Operation]:
        self.trial_closed = True
        return [searcher.Shutdown()]

    def progress(self, searcher_state: searcher.SearcherState) -> float:
        if self.trial_closed:
            return 1.0
        (the_trial,) = searcher_state.trials_created
        return searcher_state.trial_progress[the_trial] / self.max_length

    def on_trial_exited_early(
        self, _: searcher.SearcherState, request_id: uuid.UUID, exited_reason: searcher.ExitedReason
    ) -> List[searcher.Operation]:
        logging.warning(f"Trial {request_id} exited early: {exited_reason}")
        return [searcher.Shutdown()]

    def initial_operations(self, _: searcher.SearcherState) -> List[searcher.Operation]:
        logging.info("initial_operations")

        create = searcher.Create(
            request_id=uuid.uuid4(),
            hparams=self.hyperparameters,
            checkpoint=None,
        )
        validate_after = searcher.ValidateAfter(
            request_id=create.request_id, length=self.max_length
        )
        close = searcher.Close(request_id=create.request_id)
        logging.debug(f"Create({create.request_id}, {create.hparams})")
        return [create, validate_after, close]
