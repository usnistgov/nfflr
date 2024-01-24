import ase
import numpy as np

import nfflr


JVASP_64906 = {
    "cell": [
        [-1.833590720595598, 1.833590720595598, 3.4849681632445244],
        [1.833590720595598, -1.833590720595598, 3.4849681632445244],
        [1.833590720595598, 1.833590720595598, -3.4849681632445244],
    ],
    "coords": [
        [0.0, 0.0, 0.0],
        [1.83359, 0.0, 1.742485],
        [0.0, 0.0, 3.48497],
        [0.0, 1.83359, 1.742485],
    ],
    "numbers": [4, 4, 76, 44],
}


def test_fractional_coordinates():
    """nfflr.Atoms scaled_positions should match ase for non-orthorhombic cells."""
    j = JVASP_64906
    at = nfflr.Atoms(j["cell"], j["coords"], j["numbers"])
    ase_at = ase.Atoms(cell=j["cell"], positions=j["coords"], numbers=j["numbers"])

    assert np.allclose(at.scaled_positions(), ase_at.get_scaled_positions(), atol=1e-7)
