from typing import Any, List, Tuple  # noqa: D100

import gpytorch
import torch

from src.capability import Capability


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPModel(gpytorch.models.ExactGP):
    """A Gaussian Process regression model using an RBF kernel."""

    def __init__(self, train_x, train_y, likelihood, input_dim):
        super().__init__(train_x.to(device), train_y.to(device), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
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
        x_train,
        y_train,
        acquisition_function,
        num_gp_train_iterations=50,
        optimizer_lr=0.1,
    ) -> None:
        """Initialize the LBO parameters."""
        # x_train shape is [N, d].
        self.input_dim = x_train.shape[1]
        self.x_train = x_train.clone().to(device)
        self.y_train = y_train.clone().to(device)
        self.acquisition_function = acquisition_function
        self.num_gp_train_iterations = num_gp_train_iterations
        self.optimizer_lr = optimizer_lr
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = self.likelihood.to(device)
        self.model = self._train_gp()

    def _train_gp(self):
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

    def select_next_point(self, x_query):
        """Select the next query point from x_query."""
        x_query = x_query.to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _, st_devs = self.predict(x_query)
            if self.acquisition_function == "variance":
                idx = torch.argmax(st_devs)
            else:
                raise ValueError(
                    f"Acquisition function: {self.acquisition_function} is unsupported."
                )
        return idx, x_query[idx]

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


def generate_capability_using_lbo(
    capabilities: List[Capability],
    capability_scores: torch.Tensor,
    encoder: Any,
    pipeline_id: str = "nearest_neighbour",
    acquisition_function: str = "variance",
    decoder: Any = None,
    capabilities_pool: List[Capability] | None = None,
) -> Capability:
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
        acquisition_function (str): The acquisition function for LBO.
        decoder (Any, optional): The decoder model to decode the
            capability representation (only for pipeline_id="discover_new").
        capabilities_pool (List[Capability], optional): The pool of existing
            capabilities without subject model scores, used as a search space
            for the generated capability representation
            (only for pipeline_id="nearest_neighbour").

    Returns
    -------
        Capability: The generated capability.
    """
    capability_scores = capability_scores.to(device)
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

    # TODO: Part or all of the following code must be moved to run.py, especially the
    # loop on selecting the next capapbility. I'm commenting this out.
    # if pipeline_id == "nearest_neighbour":
    #     capabilities_encoding = torch.stack(
    #         [cap.encode(encoder) for cap in capabilities]
    #     )
    #     capabilities_pool_encoding = torch.stack(
    #         [cap.encode(encoder) for cap in capabilities_pool]
    #     )
    #     lbo = LBO(
    #         capabilities_encoding,
    #         capability_scores,
    #         acquisition_function,
    #     )
    #     init_pool_size = len(capabilities_pool)
    #     for _ in range(init_pool_size):
    #         idx, selected_capability_encoding = lbo.select_next_point(
    #             capabilities_pool_encoding
    #         )
    #         # TODO: Implement and call `evaluate_capability` for the selected
    #           capability to calculate its score.
    #         selected_capability_score = evaluate_capability(capabilities_pool[idx])
    #         # Remove the selected capability and its encoding.
    #         capabilities_pool.pop(idx)
    #         capabilities_pool_encoding = torch.cat(
    #             [
    #                 capabilities_pool_encoding[:idx],
    #                 capabilities_pool_encoding[idx + 1 :],
    #             ],
    #             dim=0,
    #         )
    #         lbo.update(selected_capability_encoding, selected_capability_score)
    # else:
    #     raise ValueError(f"Unsupported pipeline id: {pipeline_id}")


def generate_new_capability(
    capabilities: List[Capability],
    subject_llm_name: str,
    capabilities_pool: List[Capability] | None = None,
    **kwargs: Any,
) -> Capability:
    """
    Generate a new capability.

    Args
    ----
        capabilities (List[Capability]): The list of existing capabilities.
        subject_llm_name (str): The subject LLM model name.
        capabilities_pool (List[Capability], optional): The list of existing
            capabilities without subject model scores, used as a search space
            for the generated capability representation
            (only for pipeline_id="nearest_neighbour").

    Returns
    -------
        Capability: The generated capability.
    """
    if kwargs.get("lbo_run_id", 0) == 0:
        # Load subject LLM scores for each capability
        capability_scores = torch.Tensor(
            [cap.load_scores()[subject_llm_name] for cap in capabilities], device=device
        )
    else:
        # Only load newly added capability's score
        capability_scores = torch.Tensor(
            [capabilities[-1].load_scores()[subject_llm_name]], device=device
        )

    # TODO: Set the encoder model
    encoder = None
    if encoder is not None:
        encoder = encoder.to(device)

    pipeline_id = kwargs.get("pipeline_id", "nearest_neighbour")
    if pipeline_id == "nearest_neighbour":
        assert (
            capabilities_pool is not None
        ), "Pool of existing capabilities is not provided"
        decoder = None
    elif pipeline_id == "discover_new":
        # TODO: Set the decoder model
        decoder = None
    else:
        raise ValueError(
            f"Invalid pipeline_id: {pipeline_id}. Use either 'nearest_neighbour' or 'discover_new'."
        )

    return generate_capability_using_lbo(
        capabilities=capabilities,
        capability_scores=capability_scores,
        encoder=encoder,
        pipeline_id=pipeline_id,
        decoder=decoder,
        capabilities_pool=capabilities_pool,
    )
