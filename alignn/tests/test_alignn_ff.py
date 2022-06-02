"""Testing code for force-field related modules."""
from pathlib import Path

import pandas as pd
import torch
from jarvis.core.atoms import Atoms as JAtoms
from jarvis.db.figshare import get_jid_data
from torch.utils.data import DataLoader

from alignn.alignn_ff import ALIGNNFF_Calculator, OptimizeAtoms, RunMD
from alignn.dataset import AtomisticConfigurationDataset
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

# load example force-field data to share across tests
example_data = Path("alignn/examples/sample_data")
dataset = AtomisticConfigurationDataset(
    pd.read_json(example_data / "id_prop.json"),
    line_graph=True,
)

dl = DataLoader(dataset, collate_fn=dataset.collate, batch_size=4)


def test_force_grad():
    """Test force gradient computation.

    This triggers a segmentation fault in dgl>=0.7.2
    """
    model_cfg = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=1,
        calculate_gradient=True,
    )
    model = ALIGNNAtomWise(model_cfg)

    g, lg, t = next(iter(dl))
    out = model((g, lg))

    loss = torch.abs(out["total_energy"] - t["total_energy"]).sum()

    # should not segfault
    loss.backward()


"""
def test_alignnff():
    atoms = JAtoms.from_dict(get_jid_data()["atoms"])
    print(atoms)
    model_path = (
        "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa/out/best_model.pt"
    )
    calculator = ALIGNNFF_Calculator(model_path=model_path)
    opt = OptimizeAtoms(
        optimizer="BFGS", calculator=calculator, jarvis_atoms=atoms
    ).optimize()
    print(opt)
    print("NVT")
    md = RunMD(calculator=calculator, jarvis_atoms=opt, nsteps=5).run()
    print("NPT")
    md = RunMD(
        calculator=calculator, ensemble="npt", jarvis_atoms=opt, nsteps=5
    ).run()


# test_alignnff()
"""
