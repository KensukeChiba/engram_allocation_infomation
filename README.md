# engram_allocation_infomation
# Information-Theoretic Model of Neural Representations

This repository contains the numerical simulation code and figure generation scripts for the study on the optimization of neural representations (engram cell fractions) under context and spatial information constraints.

> **Article Title**: [Space versus Context: Competition for limited neural resources determines engram cell allocation in the hippocampus]  
> **Authors**: [Kensuke Chiba and Jun-nosuke Teramae]  
> *Submitted to / Published in PNAS.*

---

## Overview

This project investigates how neural populations optimize the allocation of engram cell fractions ($f$) across multiple contexts or spatial variables by maximizing total mutual information ($I_\mathrm{total}$).

Key functionalities provided in this code repository:
- **Single Context Analysis**: Root-finding for optimal cell fraction across occurrence probabilities $p$.
- **Multi-Context Optimization**: Sequential Least Squares Programming (`SLSQP`) to solve high-dimensional information maximization under budget constraints ($\sum f_i \le 1$).
- **Phase Boundary Calculation**: Numerical verification of analytical derivation of phase transition boundary.

---

## Repository Structure

```text
.
├── README.md                     # Project overview and instruction guide
├── LICENSE                       # License information (MIT License)
├── requirements.txt              # Required Python packages
│
├── src/                          # Main scripts for numerical experiments and figures
│   ├── no_competition.py                # Single-context optimization & visualization for Supporting Information
│   ├── phase_boundary.py         # Phase diagram generation on alpha-W plane for figure 4 (a)
│   ├── phase_boundary_fixed_alpha.py # Detailed phase boundary checks with fixed alpha for figure 4 (b)
│   ├── solve_opt.py              # Multi-context optimization with MLE confidence intervals
│   └── solve_opt_multiplot.py      # Multi-panel optimization figure generation
│
└── figure/                       # Output directory for generated PDF/PNG figures