from dataclasses import dataclass

import ase
from ase.calculators.calculator import Calculator

import torch
import einops
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ForceFieldOutput:
    """Batched force field quantities.

    energy: array of total energy per configuration
    forces: (n_atoms, 3) array of batched atomic forces
    force_ps: indices for unpacking force array with einops.unpack
    """

    energy: torch.Tensor
    forces: torch.Tensor
    force_ps: torch.Tensor

    def unpack_forces(self):
        return einops.unpack(self.forces, self.force_ps, "* d")


def collect_results(dataset, ids, model):

    e_ref, e_pred = [], []
    f_ref, f_pred = [], []
    for _id in ids:
        inputs, reference = dataset[_id]
        if isinstance(model, Calculator):
            inputs.calc = model
            energy = inputs.get_total_energy()
            forces = torch.asarray(inputs.get_forces())
        else:
            y = model(inputs)
            energy = y["energy"].item()
            forces = y["forces"].detach()

        e_ref.append(reference["energy"].item())
        e_pred.append(energy)

        f_ref.append(reference["forces"].detach())
        f_pred.append(forces)

    e_ref = torch.asarray(e_ref)
    e_pred = torch.asarray(e_pred)

    f_ref, f_ps = einops.pack(f_ref, "* d")
    f_pred, f_ps = einops.pack(f_pred, "* d")

    ref = ForceFieldOutput(e_ref, f_ref, f_ps)
    pred = ForceFieldOutput(e_pred, f_pred, f_ps)

    return ref, pred


def ecdf_plot(
    reference,
    predictions,
    quantiles=[0.5, 0.8, 0.95],
    annotate=True,
    label=None,
    axis=None,
):
    if axis is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axis

    _y = np.arange(0, 101)
    ecdf = np.percentile((reference - predictions).abs(), _y)

    ax.plot(ecdf, _y / 100, label=label)
    ax.set_ylim(0, 1.0)

    if not annotate:
        return ax

    for q in quantiles:
        v = np.percentile((reference - predictions).abs(), 100 * q)
        ax.plot([1e-16, v], [q, q], linestyle="--", color="k")
        ax.annotate(f"{v:.03f}", xy=(1.3 * v, q), va="center")

    return ax


def pseudolog10(x):
    """Signed pseudologarithm for evaluating force prediction distributions.

    for |x| large enough, 2 * sinh(x) -> sign(x) exp(|x|)

    See https://win-vector.com/2012/03/01/modeling-trick-the-signed-pseudo-logarithm/
    """
    return torch.asinh(x / 2) / np.log(10)
