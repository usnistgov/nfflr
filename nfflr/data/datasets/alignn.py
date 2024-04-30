from typing import Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from jarvis.core.atoms import Atoms as jAtoms

import nfflr
import nfflr.data.dataset


def alignn_ff_dataset(
    force_threshold: float = 10.0,
    distance_threshold: float = 5.0,
    transform: Optional[torch.nn.Module] = None,
    diskcache: bool = False,
):
    """Load filtered ALIGNNFF dataset

    force_threshold: max force component (eV/Å)
    distance_threshold: minimum nearest neighbor distance (Å)
    """
    # data = nfflr.data.dataset._load_dataset("alignn_ff_db")
    data = nfflr.data._load_dataset("alignn_ff_db")

    # store total energy (figshare provides energy per atom)
    n = data.forces.apply(len)
    data = data.assign(total_energy=n * data["total_energy"])

    # throw away configurations with max force component larger than threshold
    maxforce = data.forces.apply(lambda x: np.abs(np.array(x)).max()).values
    data = data[maxforce < force_threshold]

    # also, filter out any crystals that fails a radius graph construction
    # note: this pasted code might not work out of the box
    print("screening radius graph threshold...")
    passes, fails = [], []
    transform = nfflr.nn.PeriodicRadiusGraph(distance_threshold)
    for idx, at in tqdm(enumerate(data.atoms)):
        try:
            transform(nfflr.Atoms(jAtoms.from_dict(at)))
            passes.append(idx)
        except:
            fails.append(idx)

    print("radius graph screening complete")
    data = data.iloc[passes]

    return nfflr.AtomsDataset(
        data,
        id_tag="jid",
        group_ids=True,
        target="energy_and_forces",
        energy_units="eV",
        transform=transform,
        diskcache=diskcache,
    )
