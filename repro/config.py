"""Experiment scale.

Every experiment node runs the *same* command (``bash run.sh``); scale and
sweep design are varied here, in committed code, never through the command line
or environment variables.
"""

from __future__ import annotations

import os

# Global deterministic seed.  Every verifier derives its own child seeds from it.
SEED = 20260726

# ``full`` is the configuration reported in the release; ``smoke`` exists only
# so the suite can be exercised quickly during development.
PROFILE = os.environ.get("REPRO_PROFILE", "full")

FULL = dict(
    # -- Claim 1 ----------------------------------------------------------
    c1_symbolic_n=(2, 3, 4, 5, 6),

    # -- Claim 2 ----------------------------------------------------------
    c2_dims=(3, 4, 5, 6, 8, 10),
    c2_trials=20000,
    c2_identity_trials=2000,

    # -- Claim 3 ----------------------------------------------------------
    c3_eps=(0.20, 0.10, 0.05, 0.025),
    c3_delta=(0.40, 0.20, 0.08, 0.03),
    c3_delta_eps=0.15,
    c3_delta_repeats=3000,
    c3_dims=(3, 4, 5, 6, 8, 10),
    c3_repeats=400,          # draws used to estimate P(error <= eps) at each N
    c3_models=12,            # distinct ground-truth models per configuration

    # -- Claim 4 ----------------------------------------------------------
    # Figure 2: three panels, each sweeping one parameter.
    c4_sigmas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    c4_panels=(
        ("m", (0, 1, 3, 5), dict(n=10, p=0.5)),
        ("n", (5, 7, 10, 15), dict(m=3, p=0.5)),
        ("p", (0.2, 0.4, 0.6, 0.8), dict(n=10, m=3)),
    ),
    c4_models=50,            # paper: 50 draws of different causal models
    c4_noise=20,             # paper: 20 draws of the random noise per model

    # -- Claim 5 ----------------------------------------------------------
    c5_exhaustive_n=(3, 4),  # every acyclic digraph on n vertices
    c5_sampled_n=(5,),
    c5_samples=300,

    # -- Claim 6 ----------------------------------------------------------
    c6_exhaustive_n=(3, 4),  # every statement graph on n vertices
    c6_sampled_n=(5,),
    c6_samples=400,
    # Figure 5: three panels, sweeping injected error count.
    c6_errors=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    c6_panels=(
        ("m", (0, 1, 3, 5), dict(n=10, p=0.3)),
        ("n", (5, 7, 10, 15), dict(m=3, p=0.3)),
        ("p", (0.2, 0.3, 0.5, 0.7), dict(n=10, m=3)),
    ),
    c6_reps=50,
)

SMOKE = dict(
    FULL,
    c1_symbolic_n=(2, 3, 4),
    c2_dims=(3, 4), c2_trials=400, c2_identity_trials=100,
    c3_eps=(0.2, 0.1, 0.05), c3_delta=(0.4, 0.2, 0.08), c3_delta_eps=0.15, c3_delta_repeats=400, c3_dims=(3, 4, 5), c3_repeats=80, c3_models=2,
    c4_sigmas=(0.0, 0.2, 1.0), c4_models=4, c4_noise=3,
    c4_panels=(("m", (0, 3), dict(n=6, p=0.5)),),
    c5_exhaustive_n=(3,), c5_sampled_n=(), c5_samples=20,
    c6_exhaustive_n=(3,), c6_sampled_n=(), c6_samples=20,
    c6_errors=(0, 2, 4), c6_reps=4,
    c6_panels=(("m", (0, 3), dict(n=6, p=0.3)),),
)

CFG = FULL if PROFILE == "full" else SMOKE
