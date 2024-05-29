import lammps
import lammps.mliap
import torch
import nfflr
from nfflr.extern.lammps import MLIAPModel

lmp = lammps.lammps(cmdargs=["-echo", "both"])
lammps.mliap.activate_mliappy(lmp)

# before loading:
lmp.commands_string(
    """# 3d Lennard-Jones melt

units       lj
atom_style  atomic

lattice     fcc 0.8442
region      box block 0 1.69 0 1.69 0 1.69
create_box  1 box
create_atoms    1 box
mass        1 1.0

velocity    all create 3.0 87287 loop geom
"""
)

# unified = MLIAPModel(["Ar"], lmp=lmp)
print("python class")
torch.set_default_dtype(torch.float32)
cfg = nfflr.models.ALIGNNFFConfig(reduce_forces=False)
gnn = nfflr.models.ALIGNNFF(cfg)
model = MLIAPModel(["Ga", "N", "Si", "Al"], gnn, lmp=lmp)
# Connect the model to the mliap unified pair style.
print("lammps load")
lammps.mliap.load_unified(model)

# Run the simulation with the mliap unified pair style
lmp.commands_string(
    """
# Use pre-loaded model by specifying model filename as "EXISTS"
pair_style  mliap unified EXISTS
pair_coeff  * * Ga N Si Al

neighbor    0.3 bin
neigh_modify    every 20 delay 0 check no

fix     1 all nve

thermo      50
run     250
"""
)

lmp.close()
lmp.finalize()
