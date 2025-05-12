"""Latent Bayesian Optimization (LBO) for capability selection."""

import logging
from typing import Any, Dict, List, Tuple

import gpytorch
import torch

from src.capability import Capability


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class GPModel(gpytorch.models.ExactGP):  # type: ignore
    """A Gaussian Process regression model using an RBF kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        input_dim: int,
    ):
        super().__init__(train_x.to(device), train_y.to(device), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Compute the GP prior/posterior distribution at input x.

        Args:
            x (torch.Tensor): A tensor of input points at which to evaluate the GP.
                Shape: (n_samples, input_dim)

        Returns
        -------
            gpytorch.distributions.MultivariateNormal: A multivariate normal
            distribution representing the GP's belief over the latent function
            values at the input points `x`, characterized by the predicted mean
            and covariance.
        """
        x = x.to(device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LBO:
    """A class used to represent the Latent Bayesian Optimization (LBO) model.

    The current implementation works with a finite set of candidate points for active
    learning. In the future we will change that to support active choice of query
    points.
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        acquisition_function: str,
        num_gp_train_iterations: int = 50,
        optimizer_lr: float = 0.1,
        num_grid_points: int = 100,
        expansion_factor: float = 0.5,
    ):
        """Initialize the LBO parameters."""
        # x_train shape is [N, d].
        self.input_dim = x_train.shape[1]
        self.x_train = x_train.clone().to(device)
        self.y_train = y_train.clone().to(device)
        self.acquisition_function = acquisition_function
        self.num_gp_train_iterations = num_gp_train_iterations
        self.optimizer_lr = optimizer_lr
        self.num_grid_points = num_grid_points
        self.expansion_factor = expansion_factor
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = self.likelihood.to(device)
        self.model = self._train_gp()

    def _train_gp(self) -> GPModel:
        model = GPModel(self.x_train, self.y_train, self.likelihood, self.input_dim)
        model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.optimizer_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

        for _ in range(self.num_gp_train_iterations):
            optimizer.zero_grad()
            output = model(self.x_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        self.likelihood.eval()
        return model

    def select_next_point(
        self, x_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the next query point from x_query."""
        x_query = x_query.to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _, st_devs = self.predict(x_query)
            if self.acquisition_function == "variance":
                idx = torch.argmax(st_devs)
            elif self.acquisition_function == "expected_variance_reduction":
                total_var_reduction = []
                n_query = x_query.shape[0]
                for i in range(n_query):
                    x_star = x_query[i].unsqueeze(0)

                    # Compute covariances
                    k_star_val = self.model.covar_module(
                        x_star, x_query
                    ).evaluate()  # (1, n_query)
                    k_xx = (
                        self.model.covar_module(x_star).evaluate().squeeze()
                    )  # scalar
                    noise = self.model.likelihood.noise.item()

                    # Reduction per validation point (shape (1, n_query))
                    reduction = (k_star_val**2) / (k_xx + noise)
                    sum_reduction = reduction.sum().item()
                    total_var_reduction.append(sum_reduction)

                idx = torch.argmax(torch.tensor(total_var_reduction)).item()
            else:
                raise ValueError(
                    f"Acquisition function: {self.acquisition_function} is unsupported."
                )
        return idx, x_query[idx]

    def _create_search_grid(self) -> torch.Tensor:
        """Create a grid of points covering the expanded space around training data."""
        # Calculate bounds with expansion.
        min_vals = self.x_train.min(dim=0).values
        max_vals = self.x_train.max(dim=0).values
        ranges = max_vals - min_vals

        # Expand the bounds.
        expanded_min = min_vals - self.expansion_factor * ranges
        expanded_max = max_vals + self.expansion_factor * ranges

        # Create grid points.
        grid_points_per_dim = round(self.num_grid_points ** (1.0 / self.input_dim))
        dim_grids = [
            torch.linspace(
                expanded_min[i].item(), expanded_max[i].item(), grid_points_per_dim
            )
            for i in range(self.input_dim)
        ]

        # Create full grid using meshgrid.
        mesh = torch.meshgrid(*dim_grids)
        grid_points = torch.stack([m.flatten() for m in mesh], dim=1)

        return grid_points.to(device)

    def select_k_points(self, k: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Select k query points by finding high-variance regions in the input space.

        Args
        ----
            k: Number of points to select

        Returns
        -------
            Tuple containing:
                - List of indices of selected points in x_train
                - List of the selected points
        """
        grid_points = self._create_search_grid()
        # On the grid, find the point with highest predictive variance.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _, point = self.select_next_point(grid_points)

        # Find and return k nearest indices.
        distances = torch.norm(self.x_train - point, dim=1)
        _, topk_indices = torch.topk(distances, k, largest=False)

        selected_points = [self.x_train[i] for i in topk_indices]
        return topk_indices.tolist(), selected_points

    def update(self, q_x: torch.Tensor, q_y: torch.Tensor) -> None:
        """
        LBO update function.

        Update the training set, the query set, and the LBO model.

        Args
        ----
            q_x (torch.Tensor): The new capability representation tensor, shape (D,).
            q_y (torch.Tensor): The subject model score corresponding to q_x, shape
            (1,).

        Returns
        -------
            None
        """
        q_x = q_x.to(device)
        q_y = (
            torch.tensor([q_y], device=device)
            if not isinstance(q_y, torch.Tensor)
            else q_y.to(device)
        )
        self.x_train = torch.cat([self.x_train, q_x.unsqueeze(0)], dim=0)
        self.y_train = torch.cat([self.y_train, q_y], dim=0)
        self.model = self._train_gp()

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LBO predict function.

        Predict the scores for the given capability representations.

        Args
        ----
            x (torch.Tensor): The capability representation tensor with shape (Nc, D).

        Returns
        -------
            mean: Predicted mean values for input x.
            std: Predicted standard deviation values for input x.
        """
        x = x.to(device)
        vals = self.model(x)
        return vals.mean, vals.variance.sqrt()

    def get_error(
        self, x_test: torch.Tensor, y_test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LBO error function.

        Calculate the error between the predicted and actual scores.

        Args
        ----
            x_test (torch.Tensor): The test capability representation tensor
                with shape (Nc, D).
            y_test (torch.Tensor): The actual scores for the test capabilities.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the error.
        """
        raise NotImplementedError(
            "Error calculation is not implemented yet. "
            "Please implement the error calculation logic."
        )


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
) -> Capability:
    """
    Decode the capability representation using the decoder model.

    Args
    ----
        representation (torch.Tensor): The capability representation tensor, shape (D,).
        decoder (Any): The decoder model to decode the capability representation.

    Returns
    -------
        Capability: The decoded capability.
    """
    raise NotImplementedError


def _get_nearest_capability(
    representation: torch.Tensor,
    capabilities_pool: List[Capability],
) -> Capability:
    """
    Get the nearest capability from the existing capability pool.

    Used for selecting the capability in LBO pipeline 1.

    Args
    ----
        representation (torch.Tensor): The latent representation tensor, shape (D,).
        capabilities_pool (List[Capability]): The pool of existing capabilities.

    Returns
    -------
        Capability: The nearest capability.
    """
    raise NotImplementedError


def fit_lbo(
    capabilities: List[Capability],
    embedding_name: str,
    subject_llm_name: str,
    acquisition_function: str = "expected_variance_reduction",
) -> LBO:
    """
    Fit the Latent Bayesian Optimization (LBO) model using the existing capabilities.

    Args
    ----
        capabilities (List[Capability]): The list of existing capabilities
            used to train the LBO model.
        embedding_name (str): The name of the embedding used to represent capabilities.
        subject_llm_name (str): The name of the subject LLM used
            to evaluate capabilities.
        acquisition_function (str, optional): The acquisition function for LBO.
            Defaults to "expected_variance_reduction".

    Returns
    -------
        LBO: The fitted LBO model.
    """
    # Get the capability embeddings both for the existing capabilities
    # and the capabilities pool
    capabilities_encoding = torch.stack(
        [cap.get_embedding(embedding_name) for cap in capabilities]
    )

    # Load subject LLM scores for each existing capability
    capability_scores = torch.Tensor(
        [cap.scores[subject_llm_name]["mean"] for cap in capabilities]
    )

    # Fit the LBO model using the existing capabilities and their scores
    return LBO(
        capabilities_encoding,
        capability_scores,
        acquisition_function,
    )


def calculate_lbo_error(
    lbo_model: LBO,
    capabilities: List[Capability],
    embedding_name: str,
    subject_llm_name: str,
) -> Tuple[float, float]:
    """
    Calculate the error between the predicted and actual scores for the capabilities.

    Args
    ----
        lbo_model (LBO): The fitted LBO model.
        capabilities (List[Capability]): The list of existing capabilities
            used to train the LBO model.
        embedding_name (str): The name of the embedding used to represent capabilities.
        subject_llm_name (str): The name of the subject LLM used
            to evaluate capabilities.

    Returns
    -------
        Tuple[float, float]: RMSE and average standard deviation of candidate
        capabilities.
    """
    # Get the capability embeddings
    capabilities_encoding = torch.stack(
        [cap.get_embedding(embedding_name) for cap in capabilities]
    )

    # Load subject LLM scores for each existing capability
    capability_scores = torch.Tensor(
        [cap.scores[subject_llm_name]["mean"] for cap in capabilities]
    )

    preds_mean, preds_std = lbo_model.predict(capabilities_encoding)
    rmse = torch.sqrt(torch.mean((preds_mean - capability_scores) ** 2)).item()
    avg_std = torch.mean(preds_std).item()

    return rmse, avg_std


def select_capabilities_using_lbo(
    capabilities: List[Capability],
    embedding_name: str,
    capabilities_pool: List[Capability],
    test_capabilities: List[Capability],
    subject_llm_name: str,
    acquisition_function: str = "expected_variance_reduction",
    num_lbo_iterations: int | None = None,
) -> Tuple[List[Capability], Dict[str, List[float]]]:
    """
    Select capabilities using the Latent Bayesian Optimization (LBO) method.

    Args
    ----
        capabilities (List[Capability]): The list of existing capabilities
            used to train the LBO model.
        embedding_name (str): The name of the embedding used to represent capabilities.
        capabilities_pool (List[Capability]): The pool of candidate capabilities
            to select from during the LBO process.
        test_capabilities (List[Capability]): The list of capabilities
            used to calculate the LBO test error.
        subject_llm_name (str): The name of the subject LLM used
            to evaluate capabilities.
        acquisition_function (str, optional): The acquisition function for LBO.
            Defaults to "expected_variance_reduction".
        num_lbo_iterations (int, optional): The number of iterations
            to run the LBO process. If not provided, it defaults to the size
            of the capabilities pool.

    Returns
    -------
        List[Capability]: A list of selected capabilities generated
            using the LBO method.
    """
    if num_lbo_iterations is None:
        logger.info(
            "Number of LBO iterations is not provided. "
            "Setting it to the number of capabilities in the pool."
        )
        num_lbo_iterations = len(capabilities_pool)

    lbo = fit_lbo(
        capabilities=capabilities,
        embedding_name=embedding_name,
        subject_llm_name=subject_llm_name,
        acquisition_function=acquisition_function,
    )

    error_dict: Dict[str, List[float]] = {"rmse": [], "avg_std": []}
    # Get initial test error.
    rmse, avg_std = calculate_lbo_error(
        lbo_model=lbo,
        capabilities=test_capabilities,
        embedding_name=embedding_name,
        subject_llm_name=subject_llm_name,
    )
    error_dict["rmse"].append(rmse)
    error_dict["avg_std"].append(avg_std)

    capabilities_pool_encoding = torch.stack(
        [cap.get_embedding(embedding_name) for cap in capabilities_pool]
    )

    selected_capabilities = []
    for iter_idx in range(num_lbo_iterations):
        # Select the next capability using LBO
        idx, selected_capability_encoding = lbo.select_next_point(
            capabilities_pool_encoding
        )
        selected_capabilities.append(capabilities_pool[idx])
        logger.info(
            f"Iteration {iter_idx + 1}/{num_lbo_iterations}: "
            f"Selected capability {capabilities_pool[idx].name}"
        )
        # Obtain selected capability score, since scores for capabilities
        # in the pool are precomputed
        selected_capability_score = capabilities_pool[idx].scores[subject_llm_name][
            "mean"
        ]
        # Remove the selected capability and its embedding from the pool
        capabilities_pool.pop(idx)
        capabilities_pool_encoding = torch.cat(
            [
                capabilities_pool_encoding[:idx],
                capabilities_pool_encoding[idx + 1 :],
            ],
            dim=0,
        )
        # Update the LBO model with the selected capability
        lbo.update(selected_capability_encoding, selected_capability_score)

        # Calculate LBO error on the test set

        rmse, avg_std = calculate_lbo_error(
            lbo_model=lbo,
            capabilities=test_capabilities,
            embedding_name=embedding_name,
            subject_llm_name=subject_llm_name,
        )
        error_dict["rmse"].append(rmse)
        error_dict["avg_std"].append(avg_std)

    return selected_capabilities, error_dict
