__all__ = "NFFLrCalculator"

import torch
from ase.calculators.calculator import Calculator, all_changes

import nfflr


class NFFLrCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):

        if properties is None:
            properties = self.implemented_properties
            self.calculate(atoms, properties, system_changes)

        outputs = self.model(nfflr.Atoms(atoms))

        self.results["energy"] = outputs["total_energy"].detach().item()
        self.results["forces"] = outputs["forces"].detach().numpy()
        self.results["stress"] = (
            outputs["stress"].detach().numpy().squeeze() / atoms.get_volume()
        )
