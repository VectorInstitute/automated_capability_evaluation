import os  # noqa: D100
import random
from typing import Any, List, Tuple

import torch

from capability import Capability
from utils.constants import BASE_ARTIFACTS_DIR


class LBO:
    """A class used to represent the Latent Bayesian Optimization (LBO) model."""

    def __init__(self) -> None:
        """Initialize the LBO parameters."""
        pass

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        LBO fit function.

        Create a mapping function from the adjusted capability representations
        to the capability scores.

        Args
        ----
            X (torch.Tensor): The capability representation tensor, shape (Nc, D).
            y (torch.Tensor): The subject model scores corresponding
                to the capabilities, shape (Nc,).

        Returns
        -------
            None
        """
        raise NotImplementedError

    def update(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        LBO update function.

        Update the LBO model with new capability representation and score.

        Args
        ----
            X (torch.Tensor): The new capability representation tensor, shape (1, D).
            y (torch.Tensor): The subject model score corresponding
                to the capability, shape (1,).

        Returns
        -------
            None
        """
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        LBO predict function.

        Predict the scores for the given capability representations.

        Args
        ----
            X (torch.Tensor): The capability representation tensor with shape (Nc, D).

        Returns
        -------
            torch.Tensor: Predicted scores for subject model.
        """
        raise NotImplementedError

    def identify_high_variance_point(self) -> torch.Tensor:
        """
        Identify the capability representation with the highest variance.

        Returns
        -------
            torch.Tensor: The capability representation with the highest variance.
        """
        raise NotImplementedError


def _get_adjusted_representation(
    capabilities: List[Capability],
    capability_scores: torch.Tensor,
    encoder: Any,
    decoder: Any,
) -> torch.Tensor:
    """
    Apply the InvBO method and adjust the capabilities' representations.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        capability_score (torch.Tensor): The subject model scores.
        encoder (Any): The encoder model to encode the capability representation.
        decoder (Any): The decoder model to decode the capability representation.

    Returns
    -------
        torch.Tensor: Adjusted capabilities' representations with shape (Nc, D).
    """
    # TODO:
    # 1. Encode the capability representation using the encoder model.
    #   capability_representations = torch.stack(
    #       [elm.encode(encoder) for elm in capabilities]
    #   )
    # 2. Apply the InvBO method to adjust the capabilities' representations.
    raise NotImplementedError


def _decode_capability(
    representation: torch.Tensor,
    decoder: Any,
) -> str:
    """
    Decode the capability representation using the decoder model.

    Args
    ----
        representation (torch.Tensor): The capability representation tensor, shape (D,).
        decoder (Any): The decoder model to decode the capability representation.

    Returns
    -------
        str: The decoded capability representation.
    """
    raise NotImplementedError


def _get_nearest_capability(
    representation: torch.Tensor,
    capabilities_pool: List[str],
) -> str:
    """
    Get the nearest capability from the existing capability pool.

    Used for selecting the capability in LBO pipeline 1.

    Args
    ----
        representation (torch.Tensor): The latent representation tensor, shape (D,).
        capabilities_pool (List[str]): The pool of existing capabilities.

    Returns
    -------
        str: The nearest capability.
    """
    raise NotImplementedError


def get_lbo_train_set(
    input_data: List[str],
    train_frac: float,
    min_train_size: int,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create LBO train partition.

    Get the train set from the input data based on the train fraction.

    Args
    ----
        input_data (List[str]): The input data.
        train_frac (float): The fraction of data to use for training.
        min_train_size (int): The minimum number of training data points.

    Returns
    -------
        List[str]: The train set.
    """
    random.seed(seed)

    # Limit fraction to 2 decimal places
    train_frac = round(train_frac, 2)
    num_decimal_places = (
        len(str(train_frac).split(".")[1]) if "." in str(train_frac) else 0
    )
    min_input_data = 10 ^ num_decimal_places
    assert len(input_data) >= min_input_data, (
        f"Insufficient input data: {len(input_data)}, "
        + f"based on the given train fraction: {train_frac}."
        + f"Need ad least {min_input_data} data points."
    )

    # TODO: Improve the train set selection method
    num_train = int(len(input_data) * train_frac)
    assert num_train >= min_train_size, (
        f"Number of train data points are less than the recommended value: {min_train_size}."
    )
    train_data = random.sample(input_data, num_train)
    rem_data = list(set(input_data) - set(train_data))
    return (train_data, rem_data)


def generate_capability_using_lbo(
    capabilities: List[Capability],
    capability_scores: torch.Tensor,
    encoder: Any,
    pipeline_id: str = "nearest_neighbour",
    decoder: Any = None,
    capabilities_pool: List[str] | None = None,
) -> str:
    """
    Generate a new capability using the LBO method.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities
            used to train/update the LBO model.
        capability_scores (torch.Tensor): The subject model scores
            for the given capabilities.
        encoder (Any): The encoder model to encode the capability representation.
        pipeline_id (str): The pipeline identifier to determine the generation method.
        decoder (Any, optional): The decoder model to decode the
            capability representation (only for pipeline_id="discover_new").
        capabilities_pool (List[str], optional): The pool of existing capabilities
            without subject model scores, used as a search space for the generated
            capability representation (only for pipeline_id="nearest_neighbour").

    Returns
    -------
        str: The generated capability str representation.
    """
    # TODO:
    # 1. Apply the InvBO method to adjust the capabilities' representations.
    #       capability_representations = _get_adjusted_representation(
    #           capabilities, capability_scores, encoder, decoder
    #       )
    # 2. Fit the LBO model using the adjusted capability representations
    #   and the subject model scores.
    #   a. Fit step: If the LBO model doesn't exist (first time),
    #      create it and fit using initial capabilities
    #       lbo = LBO()
    #       lbo.fit(capability_representations, capability_scores)
    #   b. Update step: Load existing LBO model and update with new capability
    #      representation and score
    #       assert capability_representations.shape[0] == 1,
    #       "Only one capability can be updated at a time"
    #       lbo = load_lbo_model()
    #       lbo.update(capability_representations, capability_scores)
    # 3. Identify the capability representation with the highest variance.
    #   high_variance_point = lbo.identify_high_variance_point()
    # 4. Obtain new capability by either fetching nearest capability
    #   from the existing capability pool or decoding the capability
    #   representation using the decoder model.
    #       if pipeline_id == "nearest_neighbour":
    #           generated_capability = _get_nearest_capability(
    #               high_variance_point, capabilities_pool
    #           )
    #       elif pipeline_id == "discover_new":
    #           assert decoder is not None, (
    #               "Decoder model is not provided"
    #           )
    #           generated_capability = _decode_capability(
    #               high_variance_point, decoder
    #           )
    raise NotImplementedError


def generate_new_capability(
    domain: str,
    capabilities: List[str],
    subject_llm_name: str,
    capabilities_pool: List[str] | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate a new capability.

    Args
    ----
        domain (str): The domain name.
        capabilities (List[str]): The list of existing capabilities.
        subject_llm_name (str): The subject LLM model name.

    Returns
    -------
        str: The generated capability str representation.
    """
    if "trial_run" in kwargs:
        capability_dir = os.path.join(
            BASE_ARTIFACTS_DIR,
            f"capabilities_{kwargs['run_id']}",
            domain,
        )
        os.makedirs(capability_dir, exist_ok=True)
    else:
        capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "capabilities", domain)

    if kwargs.get("lbo_run_id", 0) == 0:
        # Load initial capabilities
        capability_objs = [
            Capability(os.path.join(capability_dir, cap)) for cap in capabilities
        ]
        # Load subject LLM scores for each capability
        capability_scores = torch.Tensor(
            [cap.load_scores()[subject_llm_name] for cap in capability_objs]
        )
    else:
        # Only load newly added capability and obtain subject LLM score for it
        capability_objs = [Capability(os.path.join(capability_dir, capabilities[-1]))]
        capability_scores = torch.Tensor(
            [capability_objs[-1].load_scores()[subject_llm_name]]
        )

    # TODO: Set the encoder model
    encoder = None

    pipeline_id = kwargs.get("pipeline_id", "nearest_neighbour")
    if pipeline_id == "nearest_neighbour":
        assert capabilities_pool is not None, (
            "Pool of existing capabilities is not provided"
        )
        decoder = None
    elif pipeline_id == "discover_new":
        # TODO: Set the decoder model
        decoder = None
    else:
        raise ValueError(
            f"Invalid pipeline_id: {pipeline_id}. Use either 'nearest_neighbour' or 'discover_new'."
        )

    return generate_capability_using_lbo(
        capabilities=capability_objs,
        capability_scores=capability_scores,
        encoder=encoder,
        pipeline_id=pipeline_id,
        decoder=decoder,
        capabilities_pool=capabilities_pool,
    )
