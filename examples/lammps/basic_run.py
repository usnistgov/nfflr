import lammps

# Before defining the pair style, you must do the following:
import lammps.mliap
from nfflr.extern.lammps import MLIAPModel

# Demonstrate how to load a unified model from python.
# This is essentially the same as in.mliap.unified.lj.Ar
# except that python is the driving program, and lammps
# is in library mode.

lmp = lammps.lammps(cmdargs=["-echo", "both"])
lammps.mliap.activate_mliappy(lmp)

# before loading:
lmp.commands_string(
    """# 3d Lennard-Jones melt

units       lj
atom_style  atomic

lattice     fcc 0.8442
region      box block 0 10 0 10 0 10
create_box  1 box
create_atoms    1 box
mass        1 1.0

velocity    all create 3.0 87287 loop geom
"""
)

unified = MLIAPModel(["Ar"], lmp=lmp)
# Connect the model to the mliap unified pair style.
lammps.mliap.load_unified(unified)

# Run the simulation with the mliap unified pair style
lmp.commands_string(
    """
# Use pre-loaded model by specifying model filename as "EXISTS"
pair_style  mliap unified EXISTS
pair_coeff  * * Ar

neighbor    0.3 bin
neigh_modify    every 20 delay 0 check no

fix     1 all nve

thermo      50
run     250
"""
)

lmp.close()
lmp.finalize()
