import torch
import ignite
import warnings
from copy import deepcopy


class SWAG(torch.nn.Module):
    """[SWA-Gaussian](https://arxiv.org/abs/1902.02476) implementation

    inspired by https://github.com/wjmaddox/swa_gaussian
    and https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py
    """

    def __init__(self, model, device=None, max_n_models=20, **kwargs):
        super().__init__()

        # store a structured copy of the full model to resample parameters
        self.basemodel = deepcopy(model)
        self.max_n_models = max_n_models

        if device is not None:
            self.basemodel = self.basemodel.to(device)

        # store flat buffers for parameter mean, square_mean, cov_mat_sqrt
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )

        # store flattened parameters in buffers for inclusion in state_dict
        # unflatten after sampling with torch.nn.utils.vector_to_parameters
        paramvec = torch.nn.utils.parameters_to_vector(model.parameters())
        param_sz = paramvec.size()
        self.register_buffer("mean", paramvec.new_zeros(param_sz))
        self.register_buffer("square_mean", paramvec.new_zeros(param_sz))
        self.register_buffer("cov_mat_sqrt", paramvec.new_zeros(param_sz).unsqueeze(0))

    def forward(self, *args, **kwargs):
        return self.basemodel(*args, **kwargs)

    def avg_fn(self, mean, current):
        return mean * self.n_models.item() / (self.n_models.item() + 1.0) + current / (
            self.n_models.item() + 1.0
        )

    def update_parameters(self, model):
        """SWAG update_parameters

        following the update in https://github.com/wjmaddox/swa_gaussian

        Note: assumes buffers are static and don't change over training
        """
        device = self.mean.device
        currentparams = (
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().to(device)
        )

        if self.n_averaged == 0:
            self.mean.copy_(currentparams)
            self.square_mean.copy_(currentparams**2)

        # moving average update for params and squared params
        if self.n_averaged > 0:
            n_averaged = self.n_averaged.to(device)
            self.mean.detach().copy_(
                self.avg_fn(self.mean.detach(), currentparams, n_averaged)
            )
            self.square_mean.detach().copy_(
                self.avg_fn(self.square_mean.detach(), currentparams**2, n_averaged)
            )

        # square root of covariance matrix
        # store deviation from current running mean estimate
        cov_mat_sqrt = torch.vstack([self.cov_mat_sqrt, self.mean - currentparams])
        if self.n_averaged.item() + 1 >= self.max_n_models:
            cov_mat_sqrt = cov_mat_sqrt[1:, :]

        self.cov_mat_sqrt = cov_mat_sqrt
        self.n_averaged.add_(1)


class SWAGHandler:
    """SWAG handler for ignite-based training workflow"""

    def __init__(self, model: torch.nn.Module):

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        self.model = model
        self.swagmodel = SWAG(model)

    def _update_swag_model(self, engine: ignite.engine.Engine, name: str) -> None:
        self.swagmodel.update_parameters(self.model)

    def attach(
        self,
        engine: ignite.engine.Engine,
        name: str = "swag_handler",
        event: str | ignite.engine.Events = ignite.engine.Events.EPOCH_COMPLETED,
    ):
        """Attach SWAG handler to engine."""
        engine.add_event_handler(event, self._update_swag_model, name)
