import os  # noqa: D100
from typing import Any, List

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
            y (torch.Tensor): The candidate model scores corresponding
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
            y (torch.Tensor): The candidate model score corresponding
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
            torch.Tensor: Predicted scores for candidate model.
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


def get_adjusted_representation(
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
        capability_score (torch.Tensor): The candidate model scores.
        encoder (Any): The encoder model to encode the capability representation.
        decoder (Any): The decoder model to decode the capability representation.

    Returns
    -------
        torch.Tensor: Adjusted capabilities' representations with shape (Nc, D).
    """
    # TODO:
    # 1. Encode the capability representation using the encoder model.
    #   capability_representations = torch.stack(           # noqa: ERA001
    #       [elm.encode(encoder) for elm in capabilities]   # noqa: ERA001
    #   )                                                   # noqa: ERA001
    # 2. Apply the InvBO method to adjust the capabilities' representations.
    raise NotImplementedError


def decode_capability(
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


def generate_capability_using_lbo(
    capabilities: List[Capability],
    capability_scores: torch.Tensor,
    encoder: Any,
    decoder: Any,
) -> str:
    """
    Generate a new capability using the LBO method.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        capability_scores (torch.Tensor): The candidate model scores.

    Returns
    -------
        str: The generated capability str representation.
    """
    # TODO:
    # 1. Apply the InvBO method to adjust the capabilities' representations.
    #   capability_representations = get_adjusted_representation(   # noqa: ERA001
    #       capabilities, capability_scores, encoder, decoder
    #   )                                                           # noqa: ERA001
    # 2. Fit the LBO model using the adjusted capability representations
    #   and the candidate model scores.
    #   a. Fit step: If the LBO model doesn't exist (first time),
    #      create it and fit using initial capabilities
    #       lbo = LBO()                                                 # noqa: ERA001
    #       lbo.fit(capability_representations, capability_scores)      # noqa: ERA001
    #   b. Update step: Load existing LBO model and update with new capability
    #      representation and score
    #       assert capability_representations.shape[0] == 1,
    #       "Only one capability can be updated at a time"              # noqa: ERA001
    #       lbo = load_lbo_model()                                      # noqa: ERA001
    #       lbo.update(capability_representations, capability_scores)   # noqa: ERA001
    # 3. Identify the capability representation with the highest variance.
    #   high_variance_point = lbo.identify_high_variance_point()    # noqa: ERA001
    # 4. Decode the capability representation using the decoder model.
    #   generated_capability = decode_capability(                   # noqa: ERA001
    #       high_variance_point, decoder
    #   )                                                           # noqa: ERA001
    raise NotImplementedError


def generate_new_capability(
    domain: str,
    capabilities: List[str],
    subject_llm: str,
    **kwargs: Any,
) -> str:
    """
    Generate a new capability.

    Args
    ----
        domain (str): The domain name.
        capabilities (List[str]): The list of existing capabilities.
        subject_llm (str): The subject LLM model name.

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

    if kwargs["lbo_run_id"] == 0:
        # Load initial capabilities
        capabilities_obj = [
            Capability(os.path.join(capability_dir, cap)) for cap in capabilities
        ]
        # Load subject LLM scores for each capability
        capability_scores = torch.Tensor(
            [cap.load_scores()[subject_llm] for cap in capabilities_obj]
        )
    else:
        # Only load newly added capability and obtain subject LLM score for it
        capabilities_obj = [Capability(os.path.join(capability_dir, capabilities[-1]))]
        capability_scores = torch.Tensor(
            [capabilities_obj[-1].load_scores()[subject_llm]]
        )

    # TODO: Set the encoder and decoder models
    encoder = None
    decoder = None

    return generate_capability_using_lbo(
        capabilities_obj, capability_scores, encoder, decoder
    )
