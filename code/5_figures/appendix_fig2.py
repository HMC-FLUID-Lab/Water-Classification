#!/usr/bin/env python3
"""SI Figure S2: two-component Gaussian mixture benchmark, TIP4P/2005 at -20 C.

    appendix_fig2_a   per-group distributions of q, LSI, s_k and zeta
    appendix_fig2_b   (q, zeta) scatter
"""
from benchmark_figure import benchmark_figure

if __name__ == "__main__":
    benchmark_figure("gmm", h_color="#018571", l_color="#a6611a", name="appendix_fig2")
