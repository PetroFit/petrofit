import os
from contextlib import contextmanager

import numpy as np 

from matplotlib import pyplot as plt 

__all__ = [
    "FIG_UNIT",
    "FIG_UNIT_MAX_WIDTH",
    "SCATTER_S",
    "_package_plot_style",
    "PETROFIT_rcParams",
]

# ------
# Params
# ------
AXS_BORDER_THICKNESS = 1
TICK_LEN = 12

SMALL_SIZE = 12
MEDIUM_SIZE = 16
LARGE_SIZE = 20

FIG_UNIT = 6
FIG_UNIT_MAX_WIDTH = 12
FIG_SIZE = np.array((FIG_UNIT, FIG_UNIT))
FIG_DPI = 100

SCATTER_S = 6
LINE_WIDTH = 2

# -----------------
# PETROFIT rcParams
# -----------------
PETROFIT_rcParams = {
    # --- fonts ---
    "font.family": "monospace",    
    "font.sans-serif": ["SF Pro", "SF UI Text", "Helvetica Neue", "Helvetica"],
    "mathtext.fontset": "stixsans",
    "font.size": SMALL_SIZE,          
    "axes.titlesize": MEDIUM_SIZE,    
    "axes.labelsize": MEDIUM_SIZE,    
    "xtick.labelsize": MEDIUM_SIZE,
    "ytick.labelsize": MEDIUM_SIZE,
    "legend.fontsize": SMALL_SIZE,
    "figure.titlesize": MEDIUM_SIZE,
    "lines.markersize": SCATTER_S,

    # --- figure & save ---
    "figure.figsize": FIG_SIZE,  # single-column width ≈ 8.9 cm
    "figure.dpi": FIG_DPI,
    "savefig.bbox": "tight",

    # --- axes, lines, ticks ---
    "axes.linewidth": AXS_BORDER_THICKNESS,
    "lines.linewidth": LINE_WIDTH,
    "xtick.major.size": TICK_LEN,
    "ytick.major.size": TICK_LEN,
    "xtick.minor.size": TICK_LEN/2,
    "ytick.minor.size": TICK_LEN/2,
    "xtick.major.width": AXS_BORDER_THICKNESS,
    "ytick.major.width": AXS_BORDER_THICKNESS,
    "xtick.minor.width": AXS_BORDER_THICKNESS,
    "ytick.minor.width": AXS_BORDER_THICKNESS,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.top": True,
    "ytick.right": True,
    
    # --- images ---
    "image.origin": "lower",
    "grid.linewidth" : AXS_BORDER_THICKNESS,

    # --- other ---
}


@contextmanager
def _package_plot_style():
    """Apply consistent styling to all package plots."""
    with plt.rc_context(PETROFIT_rcParams):
        yield
