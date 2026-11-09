"""Colourisation techniques for the complex function f(z) = artanh(z**8).

Every technique maps the sampled function to a palette position t in [0, 1];
all panels then use the same colormap (inferno), so they stay comparable.

Standard techniques, used for comparison:
    minmax_scaling, log_scaling, histogram_equalization
Own methods:
    piecewise_linear_quantiles, quantile_mesh_2d

Run `python colorization.py` to render all five side by side.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import rankdata


def f(z):
    """artanh(z**8) on the principal branch."""
    return np.arctanh(z**8)


def sample(n=560, r=1.5):
    """Evaluate f on an n x n grid covering [-r, r]^2 (row 0 is the top)."""
    x = -r + 2 * r * (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x[::-1])
    return f(X + 1j * Y)


# --- Standard techniques ---------------------------------------------------

def minmax_scaling(v):
    """Linear stretch of v from its minimum to its maximum onto [0, 1]."""
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / ((hi - lo) or 1.0)


def log_scaling(v):
    """Min-max scaling of log(1 + v)."""
    return minmax_scaling(np.log1p(v))


def histogram_equalization(v):
    """Colour each sample by its rank (empirical CDF); ties share their mean rank."""
    ranks = rankdata(v, method="average", nan_policy="omit").reshape(v.shape)
    return (ranks - 1) / (np.count_nonzero(~np.isnan(v)) - 1)


# --- Own methods -----------------------------------------------------------

def piecewise_linear_quantiles(v, n_quantiles=10):
    """Linear interpolation of v between its quantiles.

    The n_quantiles + 1 breakpoints (0 %, 10 %, ..., 100 % for deciles) map to
    0, 1/n, ..., 1, so each segment receives an equal share of the palette.
    """
    levels = np.linspace(0, 1, n_quantiles + 1)
    q = np.nanquantile(v, levels)
    return np.interp(v, q, levels)


def quantile_mesh_2d(w, n_quantiles=10):
    """Interpolate the palette over a triangulated mesh of Re/Im quantiles.

    The quantiles of Re w and Im w form a mesh grid. Unravelled, its nodes are
    assigned palette positions in order, from (quantile 0, quantile 0) at the
    start of the palette to (quantile 1, quantile 1) at its end. Values in
    between come from scipy's LinearNDInterpolator, which triangulates the
    unravelled mesh (Delaunay) and interpolates linearly on each triangle.
    """
    levels = np.linspace(0, 1, n_quantiles + 1)
    qa = np.nanquantile(w.real, levels)
    qb = np.nanquantile(w.imag, levels)
    A, B = np.meshgrid(qa, qb, indexing="ij")
    points = np.column_stack([A.ravel(), B.ravel()])
    values = np.linspace(0, 1, points.shape[0])
    interp = LinearNDInterpolator(points, values)
    return interp(w.real, w.imag)


TECHNIQUES = [
    ("Min–max scaling", "Standard", lambda w: minmax_scaling(np.abs(w))),
    ("Log scaling", "Standard", lambda w: log_scaling(np.abs(w))),
    ("Histogram equalization", "Standard", lambda w: histogram_equalization(np.abs(w))),
    ("Piecewise-linear deciles", "Own method", lambda w: piecewise_linear_quantiles(np.abs(w))),
    ("2-D decile mesh", "Own method", quantile_mesh_2d),
]


def main():
    import matplotlib.pyplot as plt

    w = sample()
    fig, axes = plt.subplots(1, len(TECHNIQUES), figsize=(4 * len(TECHNIQUES), 4.6))
    for ax, (name, kind, technique) in zip(axes, TECHNIQUES):
        ax.imshow(technique(w), cmap="inferno", vmin=0, vmax=1, extent=(-1.5, 1.5, -1.5, 1.5))
        ax.set_title(f"{name}\n({kind})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("atanh-colorization.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
