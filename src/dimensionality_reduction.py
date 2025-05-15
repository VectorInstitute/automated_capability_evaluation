"""Defines the DimensionalityReductionMethod class and its subclasses."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)


def _global_min_max(x: NDArray[Any]) -> Tuple[float, float]:
    return float(np.min(x)), float(np.max(x))


def _normalize(x: NDArray[Any], min_value: float, max_value: float) -> NDArray[Any]:
    return 2 * (x - min_value) / (max_value - min_value) - 1.0


class DimensionalityReductionMethod(ABC):
    """Class for dimensionality reduction methods.

    This class provides methods for dimensionality reduction of capability embeddings.
    """

    def __init__(
        self,
        method_name: str,
        output_dimension_size: int,
        random_seed: int,
        normalize_output: bool,
    ):
        self.method_name = method_name
        self.output_dimension_size = output_dimension_size
        self.random_seed = random_seed
        self.normalize_output = normalize_output
        # To be used for normalization.
        self.normalization_lower_bound = 0.0
        self.normalization_upper_bound = 0.0

        # Set torch random seed for reproducibility.
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    @abstractmethod
    def fit_transform(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fit the dimensionality reduction method to the data and transform the data.

        Args:
            embeddings (List[torch.Tensor]): List of tensors representing the embeddings
                to be reduced.

        Raises
        ------
            NotImplementedError: If the method is not implemented in a subclass.

        Returns
        -------
            List[torch.Tensor]: List of tensors representing the reduced embeddings.
        """
        raise NotImplementedError(
            "The fit_transform method should be implemented in subclasses."
        )

    @abstractmethod
    def transform_new_points(
        self, new_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Transform new points using the fitted dimensionality reduction method.

        Args:
            new_embeddings (List[torch.Tensor]): List of tensors representing the new
                embeddings to be transformed.

        Raises
        ------
            NotImplementedError: If the method is not implemented in a subclass.

        Returns
        -------
            List[torch.Tensor]: List of tensors representing the transformed embeddings.
        """
        raise NotImplementedError(
            "The transform_new_points method should be implemented in subclasses."
        )

    @classmethod
    def from_name(
        cls,
        method_name: str,
        output_dimension_size: int,
        random_seed: int,
        normalize_output: bool,
        tsne_perplexity: int | None = None,
    ) -> "DimensionalityReductionMethod":
        """Create an instance of a dimensionality reduction method based on its name.

        Args:
            method_name (str): The name of the dimensionality reduction method.
            output_dimension_size (int): The size of the output dimensions.
            random_seed (int): The random seed for reproducibility.
            normalize_output (bool): Whether to normalize the output embeddings.
            tsne_perplexity (int | None): The perplexity parameter for T-SNE. If None,
                the default value of 30 will be used.

        Returns
        -------
            DimensionalityReductionMethod: An instance of the specified dimensionality
                reduction method.

        Raises
        ------
            ValueError: If the method_name is not recognized.
        """
        if method_name.lower() == "t-sne":
            return Tsne(
                perplexity=tsne_perplexity,
                output_dimension_size=output_dimension_size,
                random_seed=random_seed,
                normalize_output=normalize_output,
            )
        if method_name.lower() == "cut-embeddings":
            return CutEmbeddings(
                output_dimension_size=output_dimension_size,
                random_seed=random_seed,
                normalize_output=normalize_output,
            )
        if method_name.lower() == "pca":
            return Pca(
                output_dimension_size=output_dimension_size,
                random_seed=random_seed,
                normalize_output=normalize_output,
            )
        raise ValueError(f"Unknown dimensionality reduction method: {method_name}")


class Tsne(DimensionalityReductionMethod):
    """Implements the T-SNE dimensionality reduction method."""

    def __init__(
        self,
        perplexity: int | None,
        output_dimension_size: int,
        random_seed: int,
        normalize_output: bool,
    ):
        super().__init__("t-sne", output_dimension_size, random_seed, normalize_output)
        if perplexity is None:
            self.perplexity = 30
        else:
            self.perplexity = perplexity

    def fit_transform(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fit and transform the T-SNE dimensionality reduction method to the data."""
        if len(embeddings) < self.perplexity:
            # Perplexity should always be smaller than the number of samples.
            # If the number of samples is smaller than the default perplexity
            # value, we set the perplexity to the number of samples - 2 since a
            # larger value either throws and error or is too big for the algorithm
            # to work properly.
            perplexity = len(embeddings) - 2
            logger.warning(
                f"Only {len(embeddings)} points are provided for t-SNE\
                perplexity is reduced to the number of points - 2."
            )
            self.perplexity = perplexity
        else:
            perplexity = self.perplexity
        logger.info(f"tsne perplexity is set to {perplexity}.")
        tsne = TSNE(
            n_components=self.output_dimension_size,
            perplexity=perplexity,
            random_state=self.random_seed,
        )
        # Convert embeddings to numpy array because that is what t-SNE expects.
        np_embeddings = np.array(embeddings)
        # The output of t-SNE is a numpy array, so we need to convert it back to
        # a list of tensors.
        reduced_embeddings = tsne.fit_transform(np_embeddings)
        if self.normalize_output:
            self.normalization_lower_bound, self.normalization_upper_bound = (
                _global_min_max(reduced_embeddings)
            )
            reduced_embeddings = _normalize(
                reduced_embeddings,
                min_value=self.normalization_lower_bound,
                max_value=self.normalization_upper_bound,
            )

        return [torch.Tensor(embedding) for embedding in reduced_embeddings]

    def transform_new_points(
        self, new_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """NotImplemented: Transform new points using the fitted T-SNE."""
        raise NotImplementedError("T-SNE cannot be used for transforming new points.")


class CutEmbeddings(DimensionalityReductionMethod):
    """Implements the CutEmbeddings dimensionality reduction method."""

    def __init__(
        self, output_dimension_size: int, random_seed: int, normalize_output: bool
    ):
        super().__init__(
            "cut-embeddings", output_dimension_size, random_seed, normalize_output
        )

    def fit_transform(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply the CutEmbeddings dimensionality reduction to the train data."""
        # Cut the embeddings to the desired size
        cut_embeddings = [
            embedding[: self.output_dimension_size] for embedding in embeddings
        ]

        if not self.normalize_output:
            return cut_embeddings

        # Convert to numpy for normalization
        np_embeddings = torch.stack(cut_embeddings).numpy()
        self.normalization_lower_bound, self.normalization_upper_bound = (
            _global_min_max(np_embeddings)
        )
        normalized = _normalize(
            np_embeddings,
            min_value=self.normalization_lower_bound,
            max_value=self.normalization_upper_bound,
        )
        return [torch.Tensor(x) for x in normalized]

    def transform_new_points(
        self, new_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply the CutEmbeddings dimensionality reduction to the test data."""
        # Cut the new points to the desired size
        cut_embeddings = [
            embedding[: self.output_dimension_size] for embedding in new_embeddings
        ]

        if not self.normalize_output:
            return cut_embeddings

        # Convert to numpy for normalization
        normalized_results = []
        for embedding in cut_embeddings:
            np_embedding = embedding.numpy()
            normalized = _normalize(
                np_embedding,
                min_value=self.normalization_lower_bound,
                max_value=self.normalization_upper_bound,
            )
            normalized_results.append(torch.Tensor(normalized))
        return normalized_results


class Pca(DimensionalityReductionMethod):
    """Implements the PCA dimensionality reduction method."""

    def __init__(
        self, output_dimension_size: int, random_seed: int, normalize_output: bool
    ) -> None:
        super().__init__("pca", output_dimension_size, random_seed, normalize_output)
        self.pca = PCA(n_components=output_dimension_size)

    def fit_transform(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fit and transform the PCA dimensionality reduction method to the data."""
        # Stack tensors and convert to numpy
        np_embeddings = torch.stack(embeddings).numpy()
        # Perform PCA
        self.pca.fit(np_embeddings)
        reduced_embeddings = self.pca.transform(np_embeddings)
        if self.normalize_output:
            self.normalization_lower_bound, self.normalization_upper_bound = (
                _global_min_max(reduced_embeddings)
            )
            reduced_embeddings = _normalize(
                reduced_embeddings,
                min_value=self.normalization_lower_bound,
                max_value=self.normalization_upper_bound,
            )

        # Convert back to PyTorch tensor, and return.
        return [torch.Tensor(embedding) for embedding in reduced_embeddings]

    def transform_new_points(
        self, new_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Transform new points using the fitted PCA dimensionality reduction method."""
        # Convert to numpy, transform, then back to torch
        np_embeddings = torch.stack(new_embeddings).numpy()
        reduced_embeddings = self.pca.transform(np_embeddings)
        if self.normalize_output:
            reduced_embeddings = _normalize(
                reduced_embeddings,
                min_value=self.normalization_lower_bound,
                max_value=self.normalization_upper_bound,
            )

        return [torch.Tensor(embedding) for embedding in reduced_embeddings]
