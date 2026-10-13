#!/usr/bin/env python3
"""SI Figure S3: HDBSCAN -> GMM (molecules removed by HDBSCAN in grey) benchmark, TIP4P/2005 at -20 C.

    appendix_fig3_a   per-group distributions of q, LSI, s_k and zeta
    appendix_fig3_b   (q, zeta) scatter
"""
from benchmark_figure import benchmark_figure

if __name__ == "__main__":
    benchmark_figure("hdbscan_gmm", h_color="#b386a0", l_color="#7aa794", name="appendix_fig3")
