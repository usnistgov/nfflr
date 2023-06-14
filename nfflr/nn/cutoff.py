import torch


def xplor_cutoff(r, r_onset=3.5, r_cutoff=4):
    """Apply smooth cutoff to pairwise interactions
    XPLOR smoothing function following HOOMD-blue and jax-md

    r: bond lengths
    r_onset: inner cutoff radius
    r_cutoff: cutoff radius

    inside cutoff radius, apply smooth cutoff envelope
    outside cutoff radius: hard zeros
    """
    r2 = r**2
    r2_on = r_onset**2
    r2_cut = r_cutoff**2

    # fmt: off
    smoothed = torch.where(
        r < r_cutoff,
        (r2_cut - r2) ** 2 * (r2_cut + 2 * r2 - 3 * r2_on) / (r2_cut - r2_on) ** 3,
        0,
    )
    return torch.where(r < r_onset, 1, smoothed)


class XPLOR(torch.nn.Module):
    def __init__(self, r_onset: float = 3.5, r_cutoff: float = 4):
        super().__init__()
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff

    def forward(self, r):
        return xplor_cutoff(r, self.r_onset, self.r_cutoff)
