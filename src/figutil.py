"""Saving helpers — caps DPI so no PNG exceeds the 2000px-per-side limit
imposed by Claude when reading multiple images in one conversation.

Importing this module also sets matplotlib's default font to a Times
New Roman-compatible serif, so all paper figures match the body text."""
from __future__ import annotations

import matplotlib as _mpl

_mpl.rcParams["font.family"] = "serif"
_mpl.rcParams["font.serif"] = [
    "Times New Roman",
    "Liberation Serif",
    "STIXGeneral",
    "DejaVu Serif",
]
_mpl.rcParams["mathtext.fontset"] = "stix"
_mpl.rcParams["axes.unicode_minus"] = False

MAX_PX = 1980  # leave a small margin under the 2000 hard limit


def safe_savefig(fig, path, dpi: int = 140, **kwargs) -> int:
    """fig.savefig(path) with dpi clamped so neither dim exceeds MAX_PX.
    bbox_inches="tight" can grow the image past nominal size, so cap on the
    nominal figsize and add headroom via MAX_PX.
    Returns the dpi actually used."""
    w_in, h_in = fig.get_size_inches()
    max_dim_in = max(w_in, h_in)
    capped_dpi = int(MAX_PX / max_dim_in)
    final_dpi = min(dpi, capped_dpi)
    fig.savefig(path, dpi=final_dpi, **kwargs)
    return final_dpi
