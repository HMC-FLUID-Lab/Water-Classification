#!/usr/bin/env python3
"""SI Figure S1: K-means (K = 2) benchmark, TIP4P/2005 at -20 C.

    appendix_fig1_a   per-group distributions of q, LSI, s_k and zeta
    appendix_fig1_b   (q, zeta) scatter
"""
from benchmark_figure import benchmark_figure

if __name__ == "__main__":
    benchmark_figure("kmeans", h_color="#5e3c99", l_color="#e66101", name="appendix_fig1")
