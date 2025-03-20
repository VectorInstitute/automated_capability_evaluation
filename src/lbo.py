from typing import Any, List  # noqa: D100

import torch

from capability import Capability


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
    #   lbo = LBO()                                                 # noqa: ERA001
    #   lbo.fit(capability_representations, capability_scores)      # noqa: ERA001
    # 3. Identify the capability representation with the highest variance.
    #   high_variance_point = lbo.identify_high_variance_point()    # noqa: ERA001
    # 4. Decode the capability representation using the decoder model.
    #   generated_capability = decode_capability(                   # noqa: ERA001
    #       high_variance_point, decoder
    #   )                                                           # noqa: ERA001
    raise NotImplementedError
