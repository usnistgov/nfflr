__all__ = "NFFLrCalculator"
from typing import Optional

import torch
from ase.calculators.calculator import Calculator, all_changes

import nfflr


class NFFLrCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: torch.nn.Module,
        scaler: Optional[torch.nn.Module] = None,
        log_history: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.scaler = scaler

        self.log_history = log_history
        self.history = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):

        if properties is None:
            properties = self.implemented_properties
            self.calculate(atoms, properties, system_changes)

        outputs = self.model(nfflr.Atoms(atoms))

        energy = outputs["energy"].detach()
        forces = outputs["forces"].detach()

        if self.scaler is not None:
            energy = self.scaler.inverse_transform(energy)
            forces = self.scaler.unscale(forces)

        self.results["energy"] = energy.item()
        self.results["forces"] = forces.numpy()
        self.results["stress"] = (
            outputs["stress"].detach().numpy().squeeze() / atoms.get_volume()
        )

        if self.log_history:
            self.history.append(energy.item())
